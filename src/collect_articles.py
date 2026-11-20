import json, time, requests
from src.config import path

API = "https://en.wikipedia.org/w/api.php"
# A more descriptive User-Agent is strongly recommended.
HEADERS = {
    "User-Agent": "RAG101-Research-Project/1.0 (https://github.com/Shah-Daniyal-Quadri/final-project_RAG_101; Shah.2.quadri@gamil.com) requests"
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

# The full list of 80 titles (from the previous step)
TITLES = [
    # Mughal Empire
    "Mughal Empire", "Babur", "Humayun", "Akbar", "Jahangir",
    "Shah Jahan", "Aurangzeb", "Nur Jahan", "Mumtaz Mahal", "Taj Mahal",
    "First Battle of Panipat", "Second Battle of Panipat", "Fatehpur Sikri", "Red Fort", "Din-i Ilahi",
    "Birbal", "Sher Shah Suri", "Akbarnama", "Abul Fazl", "Mughal painting",

    # Scientific Revolution
    "Scientific Revolution", "Nicolaus Copernicus", "De revolutionibus orbium coelestium", "Galileo Galilei", "Galileo affair",
    "Johannes Kepler", "Kepler's laws of planetary motion", "Tycho Brahe", "Isaac Newton", "Philosophiæ Naturalis Principia Mathematica",
    "Royal Society", "Francis Bacon", "René Descartes", "Robert Hooke", "Robert Boyle",
    "Christiaan Huygens", "Edmond Halley", "Andreas Vesalius", "William Harvey", "Gottfried Wilhelm Leibniz",

    # Mongol Empire
    "Mongol Empire", "Genghis Khan", "Ögedei Khan", "Möngke Khan", "Kublai Khan",
    "Hulagu Khan", "Batu Khan", "Yuan dynasty", "Golden Horde", "Ilkhanate",
    "Chagatai Khanate", "Siege of Baghdad (1258)", "Battle of Ain Jalut", "Mongol invasions of Japan", "Marco Polo",
    "The Travels of Marco Polo", "Pax Mongolica", "Yassa", "Börte", "Subutai",

    # Age of Exploration
    "Age of Discovery", "Prince Henry the Navigator", "Bartolomeu Dias", "Vasco da Gama", "Christopher Columbus",
    "Isabella I of Castile", "Ferdinand Magellan", "Juan Sebastián Elcano", "Magellan expedition", "Treaty of Tordesillas",
    "Amerigo Vespucci", "Pedro Álvares Cabral", "John Cabot", "Hernán Cortés", "Francisco Pizarro",
    "Fall of Tenochtitlan", "Columbian exchange", "Portuguese Empire", "Caravel", "Carrack",
]

def main():
    # Load what we already have
    try:
        with open(path("articles"), encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    by_title = {a["title"]: a for a in existing}

    with requests.Session() as session:
        for t in TITLES:
            if t in by_title:
                print(f"Cached: {t}")
                continue
            print(f"Fetching: {t}")
            article = fetch_article(session, t)
            by_title[article["title"]] = article
            time.sleep(1.0)

    out = list(by_title.values())
    path("articles").parent.mkdir(parents=True, exist_ok=True)
    with open(path("articles"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} articles -> {path('articles')}")

if __name__ == "__main__":
    main()