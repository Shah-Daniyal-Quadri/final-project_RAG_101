import json, time, requests
from src.config import path

API = "https://en.wikipedia.org/w/api.php"
# A more descriptive User-Agent is strongly recommended.
HEADERS = {
    "User-Agent": "RAG101-Research-Project/1.0 (https://your-project-url-or-email-here; contact@example.com) requests"
}

def fetch_article(session: requests.Session, title: str, max_retries: int = 5) -> dict:
    """Fetches a single article with exponential backoff for 429 errors."""
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1,
    }
    for attempt in range(max_retries):
        try:
            r = session.get(API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                # Respect the Retry-After header if provided.
                retry_after = int(r.headers.get("Retry-After", 2 ** attempt))
                print(f"Rate limited (429). Retrying in {retry_after}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(retry_after)
                continue # Try again
            r.raise_for_status() # Raise for other errors (4xx, 5xx)
            
            pages = r.json()["query"]["pages"]
            page = next(iter(pages.values()))
            return {
                "title": page["title"],
                "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
                "text": page.get("extract", ""),
            }
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e # Re-raise on the last attempt
            wait = 2 ** attempt
            print(f"Network error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception(f"Failed to fetch '{title}' after {max_retries} attempts.")

# The full list of 15 titles (from the previous step)
TITLES = [
    "Protestant Reformation", "Martin Luther", "Ninety-five Theses",
    "John Calvin", "Counter-Reformation",
    "Islamic Golden Age", "House of Wisdom", "Al-Khwarizmi", "Abbasid Caliphate",
    "Crusades", "First Crusade", "Reconquista",
    "History of Buddhism", "Silk Road", "Ashoka",
]

def main():
    # Create a session to reuse the TCP connection and handle cookies/rate-limiting better.
    with requests.Session() as session:
        out = []
        for t in TITLES:
            # Check if article was already fetched to resume progress.
            try:
                with open(path("articles"), "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if any(a["title"] == t for a in existing_data):
                        print(f"Skipping (already fetched): {t}")
                        out = existing_data # Keep existing data
                        continue
            except (FileNotFoundError, json.JSONDecodeError):
                pass # File doesn't exist or is empty, start fresh.
            
            print(f"Fetching: {t}")
            article = fetch_article(session, t)
            out.append(article)
            time.sleep(1.0) # A respectful delay between different articles.
        
        # Deduplicate in case of re-runs.
        unique_out = {a["title"]: a for a in out}.values()
        
        path("articles").parent.mkdir(parents=True, exist_ok=True)
        with open(path("articles"), "w", encoding="utf-8") as f:
            json.dump(list(unique_out), f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(unique_out)} articles → {path('articles')}")

if __name__ == "__main__":
    main()