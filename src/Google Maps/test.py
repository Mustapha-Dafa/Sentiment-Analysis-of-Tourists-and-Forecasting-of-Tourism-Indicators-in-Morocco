import os
import re
import json
import csv
import time
import math
import random
import urllib.parse
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# >>> WebDriverManager pour éviter de gérer chromedriver manuellement
from webdriver_manager.chrome import ChromeDriverManager


# ---------------------------
# CONFIG
# ---------------------------

CATEGORIES = ["Hotel"]

CITIES = [
    "Marrakech"
]

# Limites raisonnables pour éviter d'être trop agressif
MAX_PLACES_PER_QUERY = 30          # nb de lieux à collecter par (catégorie, ville)
MAX_REVIEWS_PER_PLACE = 40         # nb d'avis max par lieu
SCROLL_PAUSE_SEC = 1.25            # pause entre scrolls
GLOBAL_WAIT_SEC = 18               # temps d'attente explicite (WebDriverWait)
SLEEP_BETWEEN_PLACES = (1.0, 2.2)  # jitter sleep entre 2 lieux (min,max)
SLEEP_BETWEEN_QUERIES = 3.0        # pause entre 2 requêtes (catégorie,ville)

# Dossier de sortie
OUT_DIR = Path("output_gmaps")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLACES_CSV = OUT_DIR / "places.csv"
REVIEWS_JSONL = OUT_DIR / "reviews.jsonl"


# ---------------------------
# UTILITAIRES
# ---------------------------

def build_driver(headless: bool = True) -> webdriver.Chrome:
    chrome_options = Options()
    # Headless moderne
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1366,768")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # UA réaliste
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    # optionnel: désactiver les images pour accélérer
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def jitter_sleep(a: float, b: float):
    time.sleep(random.uniform(a, b))


def gmaps_search_url(category: str, city: str, lang: str = "fr") -> str:
    q = urllib.parse.quote_plus(f"{category} {city}")
    return f"https://www.google.com/maps/search/{q}?hl={lang}"


def get_results_feed(driver) -> Optional[webdriver.Chrome]:
    """
    Essaie de récupérer le conteneur défilant de la liste des résultats.
    """
    wait = WebDriverWait(driver, GLOBAL_WAIT_SEC)
    candidates = [
        (By.XPATH, "//div[@role='feed']"),
        (By.XPATH, "//div[contains(@aria-label, 'Résultats') or contains(@aria-label,'Results')]"),
    ]
    for how, sel in candidates:
        try:
            return wait.until(EC.presence_of_element_located((how, sel)))
        except Exception:
            pass
    return None


def scroll_element_to_load_all(driver, element, target_min_links: int = 50, max_tries: int = 25):
    """
    Scroll l'élément (colonne des résultats) pour charger plus de fiches.
    On arrête si la hauteur ou le nombre de liens n'augmente plus.
    """
    last_height = 0
    stable_rounds = 0
    for i in range(max_tries):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", element)
        time.sleep(SCROLL_PAUSE_SEC)
        new_height = driver.execute_script('return arguments[0].scrollHeight', element)
        if new_height <= last_height:
            stable_rounds += 1
        else:
            stable_rounds = 0
        last_height = new_height

        # Heuristique: si plusieurs tours stables, on stoppe
        if stable_rounds >= 2:
            break


def collect_place_links_from_feed(feed_el) -> List[str]:
    """
    Récupère les liens /maps/place/... dans la colonne de résultats.
    """
    links = []
    anchors = feed_el.find_elements(By.XPATH, ".//a[contains(@href,'/maps/place/')]")
    seen = set()
    for a in anchors:
        href = a.get_attribute("href") or ""
        if "/maps/place/" in href and href not in seen:
            links.append(href)
            seen.add(href)
    return links


def parse_lat_lng_from_url(url: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Essaye 2 formats: 
      - ...!8m2!3d<lat>!4d<lng>
      - .../@<lat>,<lng>,<zoom>...
    """
    m = re.search(r"!8m2!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)", url)
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.search(r"/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),", url)
    if m2:
        return float(m2.group(1)), float(m2.group(2))
    return None, None


def wait_and_click_reviews_tab(driver) -> bool:
    """
    Clique sur le bouton 'Avis' (FR) ou 'Reviews' (EN). Retourne True si ok.
    """
    wait = WebDriverWait(driver, GLOBAL_WAIT_SEC)
    candidates = [
        (By.XPATH, "//button[contains(@aria-label,'Avis')]"),
        (By.XPATH, "//button[contains(@aria-label,'Reviews')]"),
        (By.XPATH, "//button[.//div[text()='Avis'] or .//div[text()='Reviews']]"),
    ]
    for how, sel in candidates:
        try:
            el = wait.until(EC.element_to_be_clickable((how, sel)))
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            pass
    return False


def get_reviews_container(driver):
    """
    Retourne le conteneur scrollable des avis.
    """
    wait = WebDriverWait(driver, GLOBAL_WAIT_SEC)
    candidates = [
        (By.XPATH, "//div[contains(@class,'DxyBCb') and contains(@class,'m6QErb')]"),
        (By.XPATH, "//div[@role='feed']"),
    ]
    for how, sel in candidates:
        try:
            return wait.until(EC.presence_of_element_located((how, sel)))
        except Exception:
            pass
    return None


def expand_all_review_snippets(driver):
    """
    Clique sur tous les boutons 'Afficher plus / Plus / More' pour dérouler le texte.
    On tente plusieurs patterns (classes et textes).
    """
    # plusieurs passes (certains apparaissent après scroll)
    for _ in range(2):
        buttons = driver.find_elements(
            By.XPATH,
            "//button[contains(@class,'w8nwRe') or contains(@aria-label,'Plus') or contains(.,'Plus') or contains(.,'More')]"
        )
        for b in buttons:
            try:
                driver.execute_script("arguments[0].click();", b)
                time.sleep(0.1)
            except Exception:
                pass


def extract_place_name(driver) -> Optional[str]:
    """
    Nom du lieu, souvent dans un H1 avec classe 'DUwDvf'.
    """
    wait = WebDriverWait(driver, GLOBAL_WAIT_SEC)
    candidates = [
        (By.XPATH, "//h1[contains(@class,'DUwDvf')]"),
        (By.XPATH, "//h1"),
    ]
    for how, sel in candidates:
        try:
            el = wait.until(EC.presence_of_element_located((how, sel)))
            text = (el.text or "").strip()
            if text:
                return text
        except Exception:
            pass
    return None


def extract_rating_and_count(driver) -> Tuple[Optional[float], Optional[int]]:
    """
    Essaie de récupérer la note moyenne (sur 5) et le nombre d'avis (entier).
    """
    # 1) aria-label de type "4,5 étoiles sur 5"
    try:
        el = driver.find_element(By.XPATH, "//*[@aria-label and (contains(@aria-label,'étoiles') or contains(@aria-label,'stars'))]")
        label = el.get_attribute("aria-label") or ""
        # tenter les formats FR et EN
        m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:étoiles|stars)", label)
        rating = float(m.group(1).replace(",", ".")) if m else None
    except Exception:
        rating = None

    # 2) bouton "Avis (1 234)" ou "Reviews (1,234)"
    try:
        btn = driver.find_element(By.XPATH, "//button[contains(@aria-label,'Avis') or contains(@aria-label,'Reviews') or .//div[text()='Avis'] or .//div[text()='Reviews']]")
        txt = btn.text or ""
        digits = re.findall(r"\d+", txt.replace("\u202f", "").replace("\u00A0",""))
        count = int("".join(digits)) if digits else None
    except Exception:
        count = None

    return rating, count


def scroll_reviews_and_collect(driver, place_name: str, city: str, category: str, max_reviews: int) -> List[Dict]:
    """
    Scroll le panneau d'avis et extrait (note, date, texte) pour chaque avis.
    """
    out = []
    container = get_reviews_container(driver)
    if not container:
        return out

    # Charger progressivement jusqu'à max_reviews (ou fin)
    last_len = -1
    stable_rounds = 0
    tries = 0
    while len(out) < max_reviews and tries < 30:
        expand_all_review_snippets(driver)

        cards = container.find_elements(By.XPATH, ".//div[contains(@class, 'jftiEf')]")
        # Extraire ce qui est visible (on repart de zéro et on déduplique via tuple)
        tmp = []
        for c in cards:
            try:
                # note (aria-label "4,0 étoiles" etc.)
                try:
                    star_el = c.find_element(By.XPATH, ".//span[contains(@class,'kvMYJc')]")
                    star_label = star_el.get_attribute("aria-label") or ""
                    m = re.search(r"(-?\d+(?:[.,]\d+)?)", star_label)
                    rating = float(m.group(1).replace(",", ".")) if m else None
                except Exception:
                    rating = None

                # date (ex "il y a 2 mois")
                try:
                    date_el = c.find_element(By.XPATH, ".//span[contains(@class,'rsqaWe')]")
                    date_txt = (date_el.text or "").strip()
                except Exception:
                    date_txt = None

                # texte avis
                try:
                    review_el = c.find_element(By.XPATH, ".//span[contains(@class,'wiI7pd')]")
                    review_txt = (review_el.text or "").strip()
                except Exception:
                    review_txt = ""

                if rating is None and (not review_txt):
                    continue

                tmp.append({
                    "city": city,
                    "category": category,
                    "place_name": place_name,
                    "rating": rating,
                    "date": date_txt,
                    "review": review_txt
                })
            except Exception:
                pass

        # Dédupliquer en gardant l'ordre
        seen = set()
        dedup = []
        for r in tmp:
            key = (r["date"], r["review"])
            if key not in seen:
                dedup.append(r)
                seen.add(key)

        out = dedup  # on garde la dernière vue dédupliquée

        # si on n'a pas encore assez d'avis, on scroll encore
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", container)
        time.sleep(SCROLL_PAUSE_SEC)
        tries += 1

        if len(out) == last_len:
            stable_rounds += 1
        else:
            stable_rounds = 0
        last_len = len(out)

        if stable_rounds >= 2:
            break

    return out[:max_reviews]


def write_places_csv_header_once():
    if not PLACES_CSV.exists():
        with open(PLACES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["city","category","place_name","rating","reviews_count","lat","lng","url"])


def append_place_row(city, category, place_name, rating, reviews_count, lat, lng, url):
    with open(PLACES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([city, category, place_name, rating, reviews_count, lat, lng, url])


def append_reviews_jsonl(reviews: List[Dict]):
    if not reviews:
        return
    with open(REVIEWS_JSONL, "a", encoding="utf-8") as f:
        for r in reviews:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# PIPELINE PRINCIPAL
# ---------------------------

def collect_places_for_query(driver, category: str, city: str) -> List[str]:
    """
    Ouvre la recherche Google Maps et retourne une liste d'URLs de lieux (/maps/place/...)
    """
    url = gmaps_search_url(category, city)
    driver.get(url)

    feed = get_results_feed(driver)
    if not feed:
        print(f"[WARN] Aucun conteneur de résultats pour {category} / {city}")
        return []

    # scroll le feed pour charger un max de cartes
    scroll_element_to_load_all(driver, feed)

    links = collect_place_links_from_feed(feed)
    links = links[:MAX_PLACES_PER_QUERY]
    return links


def process_place(driver, place_url: str, city: str, category: str):
    """
    Ouvre une fiche lieu, récupère méta + avis, sauvegarde.
    """
    driver.get(place_url)
    place_name = extract_place_name(driver) or "(inconnu)"
    lat, lng = parse_lat_lng_from_url(place_url)
    rating, reviews_count = extract_rating_and_count(driver)

    # Sauvegarde ligne "lieu"
    append_place_row(city, category, place_name, rating, reviews_count, lat, lng, place_url)

    # Aller vers avis
    if wait_and_click_reviews_tab(driver):
        revs = scroll_reviews_and_collect(driver, place_name, city, category, MAX_REVIEWS_PER_PLACE)
        append_reviews_jsonl(revs)


def main():
    write_places_csv_header_once()
    driver = build_driver(headless=True)

    try:
        for category in CATEGORIES:
            for city in CITIES:
                print(f"\n=== [{category}] / [{city}] ===")
                place_links = collect_places_for_query(driver, category, city)
                print(f"[INFO] {len(place_links)} lieux trouvés pour {category} / {city}")

                for idx, link in enumerate(place_links, 1):
                    print(f"  -> ({idx}/{len(place_links)}) {link}")
                    try:
                        process_place(driver, link, city, category)
                    except Exception as e:
                        print(f"[ERR] Echec place: {e}")
                    jitter_sleep(*SLEEP_BETWEEN_PLACES)

                time.sleep(SLEEP_BETWEEN_QUERIES)

    finally:
        driver.quit()

    print(f"\n[OK] Export lieux: {PLACES_CSV}")
    print(f"[OK] Export avis : {REVIEWS_JSONL}")


if __name__ == "__main__":
    main()
