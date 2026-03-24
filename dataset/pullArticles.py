# getting the articles from wiki
import requests
import random
import pandas as pd
from bs4 import BeautifulSoup
import hashlib
from time import sleep
from tqdm import tqdm
import os

# ---------------- CONFIG ----------------
TARGET_COUNT = 2000
SAVE_EVERY = 50
MIN_PARA_LEN = 150      # relaxed
MIN_PARAS = 2           # relaxed
MAX_PARAS = 10          # relaxed
SLEEP_TIME = 1.0        # safer for API
MAX_ATTEMPTS = 20000    # prevents infinite loop

SOURCE = "Simple Wikipedia"
DOMAIN_DEFAULT = "general"
OUTPUT_FILE = "simple_wikipedia_random_domain_dataset.xlsx"
# ---------------------------------------

API_URL = "https://simple.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "KaggleDatasetBuilder/1.0 (academic use)"
}

DOMAIN_KEYWORDS = {
    "sports": ["sport", "football", "cricket", "olympic"],
    "science": ["science", "physics", "chemistry", "biology"],
    "technology": ["technology", "computer", "software"],
    "history": ["history", "war", "empire"],
    "geography": ["country", "city", "river", "mountain"],
    "politics": ["politician", "government", "election"],
    "entertainment": ["movie", "film", "music", "actor"],
    "people": ["births", "living people"]
}

# ---------------- HELPERS ----------------
def safe_get(params):
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)

        if r.status_code == 429:  # rate limit
            sleep(5)
            return None

        if r.status_code != 200 or not r.text.strip():
            return None

        return r.json()

    except Exception:
        return None


def normalize_paragraphs(paragraphs):
    merged, buffer = [], ""
    for p in paragraphs:
        if len(buffer) < MIN_PARA_LEN:
            buffer += (" " if buffer else "") + p
        else:
            merged.append(buffer)
            buffer = p
    if buffer:
        merged.append(buffer)
    return merged


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(text):
    words = sorted(set(text.lower().split()))
    return hashlib.md5(" ".join(words[:300]).encode("utf-8")).hexdigest()


def infer_domain(categories):
    joined = " ".join(categories).lower()
    for domain, keys in DOMAIN_KEYWORDS.items():
        if any(k in joined for k in keys):
            return domain
    return DOMAIN_DEFAULT


def get_random_title():
    data = safe_get({
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 1,
        "format": "json"
    })
    if not data:
        return None
    return data["query"]["random"][0]["title"]


def get_article(title):
    return safe_get({
        "action": "parse",
        "page": title,
        "prop": "text|categories",
        "format": "json"
    })


# ---------------- LOAD EXISTING DATA ----------------
if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_excel(OUTPUT_FILE)
    rows = df_existing.to_dict("records")

    seen_titles = set(df_existing["title"])
    seen_urls = set(df_existing["url"])
    seen_hashes = set(df_existing["content"].apply(hash_text))
    seen_fingerprints = set(df_existing["content"].apply(fingerprint))

    article_id = len(rows) + 1
else:
    rows = []
    seen_titles = set()
    seen_urls = set()
    seen_hashes = set()
    seen_fingerprints = set()
    article_id = 1


# ---------------- COLLECT DATA ----------------
attempts = 0
progress = tqdm(total=TARGET_COUNT, initial=len(rows), desc="Building dataset")

while len(rows) < TARGET_COUNT and attempts < MAX_ATTEMPTS:
    attempts += 1

    title = get_random_title()
    if not title or title in seen_titles:
        sleep(SLEEP_TIME)
        continue

    data = get_article(title)
    if not data or "parse" not in data:
        sleep(SLEEP_TIME)
        continue

    html = data["parse"]["text"]["*"]
    categories = [c["*"] for c in data["parse"].get("categories", [])]

    soup = BeautifulSoup(html, "html.parser")
    paras = [
        p.get_text().strip()
        for p in soup.find_all("p")
        if len(p.get_text().strip()) > 40
    ]

    merged = normalize_paragraphs(paras)

    if not (MIN_PARAS <= len(merged) <= MAX_PARAS):
        sleep(SLEEP_TIME)
        continue

    content = "\n\n".join(merged)
    h = hash_text(content)
    fp = fingerprint(content)

    if h in seen_hashes or fp in seen_fingerprints:
        sleep(SLEEP_TIME)
        continue

    url = f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"

    rows.append({
        "id": f"wiki_{article_id:04d}",
        "title": title,
        "source": SOURCE,
        "domain": infer_domain(categories),
        "content": content,
        "paragraph_count": len(merged),
        "url": url
    })

    seen_titles.add(title)
    seen_urls.add(url)
    seen_hashes.add(h)
    seen_fingerprints.add(fp)

    article_id += 1
    progress.update(1)

    # -------- SAVE EVERY 50 --------
    if len(rows) % SAVE_EVERY == 0:
        pd.DataFrame(rows).to_excel(OUTPUT_FILE, index=False)
        print(f"💾 Saved {len(rows)} articles")

    sleep(SLEEP_TIME)

progress.close()

# ---------------- FINAL SAVE ----------------
pd.DataFrame(rows).to_excel(OUTPUT_FILE, index=False)
print(f"✅ FINAL SAVE: {len(rows)} articles written to {OUTPUT_FILE}")
print(f"🧮 Total attempts: {attempts}")
