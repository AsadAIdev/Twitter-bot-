"""
AI Twitter/X Bot
-----------------
Features:
  1. AI tweet generation (Google Gemini)
  2. Topic rotation (won't repeat same topic back-to-back)
  3. Duplicate-tweet prevention (keeps local history)
  4. Auto hashtag generation
  5. Posting via Upload-Post API
  6. Retry logic with exponential backoff (handles temporary API failures)
  7. Logging (console + log file)
  8. Character-limit safety (auto-trims to 280 chars)
  9. Dry-run mode for testing without actually posting
  10. Fully configurable via .env / environment variables (safe for GitHub Actions secrets)
"""

import os
import sys
import json
import random
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "tweet_history.json"
LOG_FILE = BASE_DIR / "bot.log"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPLOAD_POST_API_KEY = os.getenv("UPLOAD_POST_API_KEY")
UPLOAD_POST_USER = os.getenv("UPLOAD_POST_USER")  # profile/user configured in Upload-Post
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)
UPLOAD_POST_URL = "https://api.upload-post.com/api/upload_text"

MAX_TWEET_LEN = 280
MAX_RETRIES = 3

TOPICS = [
    "AI aur machine learning ki dunya se ek useful tip",
    "programming/coding motivation ya productivity tip",
    "tech industry ki latest trend (general, evergreen)",
    "freelancing aur remote work ke liye advice",
    "student/self-learner ke liye study/skill-building tip",
    "an interesting tech fact people don't usually know",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("twitter-bot")


# ----------------------------------------------------------------------------
# History (prevents duplicate tweets, tracks last used topic)
# ----------------------------------------------------------------------------
def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return {"tweets": [], "last_topic": None}


def save_history(history):
    history["tweets"] = history["tweets"][-100:]  # keep last 100 only
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def pick_topic(history):
    choices = [t for t in TOPICS if t != history.get("last_topic")]
    return random.choice(choices or TOPICS)


# ----------------------------------------------------------------------------
# Tweet generation (Gemini)
# ----------------------------------------------------------------------------
def generate_tweet(topic: str) -> str:
    prompt = (
        f"Write ONE original tweet about: {topic}.\n"
        f"Rules:\n"
        f"- Under 260 characters (leave room for hashtags)\n"
        f"- No hashtags in this part (added separately)\n"
        f"- No quotation marks around the tweet\n"
        f"- Sound natural, engaging, not robotic\n"
        f"- Plain text only, one tweet, nothing else in the response"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text.strip('"').strip()
        except Exception as e:
            log.warning(f"Gemini attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Gemini API failed after all retries")


def generate_hashtags(topic: str) -> str:
    prompt = (
        f"Give 2-3 short, relevant Twitter hashtags (no explanation, just "
        f"space-separated hashtags like #AI #Tech) for a tweet about: {topic}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text
    except Exception as e:
        log.warning(f"Hashtag generation failed, skipping: {e}")
        return ""


def build_final_tweet(body: str, hashtags: str) -> str:
    tweet = body.strip()
    if hashtags:
        combined = f"{tweet}\n\n{hashtags.strip()}"
        if len(combined) <= MAX_TWEET_LEN:
            tweet = combined
    if len(tweet) > MAX_TWEET_LEN:
        tweet = tweet[: MAX_TWEET_LEN - 1].rsplit(" ", 1)[0] + "…"
    return tweet


# ----------------------------------------------------------------------------
# Posting (Upload-Post API)
# ----------------------------------------------------------------------------
def post_tweet(text: str):
    if DRY_RUN:
        log.info(f"[DRY RUN] Would post:\n{text}")
        return {"dry_run": True}

    headers = {"Authorization": f"ApiKey {UPLOAD_POST_API_KEY}"}
    data = {
        "user": UPLOAD_POST_USER,
        "platform[]": "twitter",
        "text": text,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(UPLOAD_POST_URL, headers=headers, data=data, timeout=30)
            resp.raise_for_status()
            log.info("Tweet posted successfully.")
            return resp.json()
        except Exception as e:
            log.warning(f"Post attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Upload-Post API failed after all retries")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY missing. Set it in .env or GitHub secrets.")
        sys.exit(1)
    if not DRY_RUN and (not UPLOAD_POST_API_KEY or not UPLOAD_POST_USER):
        log.error("UPLOAD_POST_API_KEY / UPLOAD_POST_USER missing.")
        sys.exit(1)

    history = load_history()
    topic = pick_topic(history)
    log.info(f"Selected topic: {topic}")

    body = generate_tweet(topic)
    hashtags = generate_hashtags(topic)
    final_tweet = build_final_tweet(body, hashtags)

    # duplicate check
    if final_tweet in history["tweets"]:
        log.info("Duplicate detected, regenerating once...")
        body = generate_tweet(topic)
        final_tweet = build_final_tweet(body, hashtags)

    log.info(f"Final tweet ({len(final_tweet)} chars):\n{final_tweet}")

    result = post_tweet(final_tweet)

    history["tweets"].append(final_tweet)
    history["last_topic"] = topic
    history["last_run"] = datetime.now(timezone.utc).isoformat()
    save_history(history)

    log.info(f"Done. Result: {result}")


if __name__ == "__main__":
    main()
