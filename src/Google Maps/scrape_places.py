from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from itertools import zip_longest
import time,random , os
import pandas as pd
import urllib.parse


def make_driver(headless=False):
    from selenium.webdriver.chrome.options import Options
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--lang=fr-FR")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(2)  # petit implicite pour réduire le code
    return driver


def build_search_url(category: str, city: str, lang: str = "fr") -> str:
    q = urllib.parse.quote_plus(f"{category} {city}")
    return f"https://www.google.com/maps/search/{q}" #?hl={lang}"

WAIT_SEC = 15
MAX_SCROLLS = 25
SCROLL_PAUSE = 1.3
WAIT_SEC = 15

def get_results_feed(driver):
    wait = WebDriverWait(driver, WAIT_SEC)
    return wait.until(EC.presence_of_element_located((By.XPATH, "//div[@role='feed']")))

def scroll_results(driver, feed):
    last_count = -1
    stable_rounds = 0
    for _ in range(MAX_SCROLLS):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", feed)
        time.sleep(SCROLL_PAUSE)
        cards = feed.find_elements(By.XPATH, ".//a[contains(@href,'/maps/place/')]")
        if len(cards) == last_count:
            stable_rounds += 1
        else:
            stable_rounds = 0
        last_count = len(cards)
        if stable_rounds >= 2:
            break


def collect_places_from_feed(feed,categoty,city):
    """Return list of (name, link)."""
    results = []
    seen = set()
    anchors = feed.find_elements(By.XPATH, ".//a[contains(@href,'/maps/place/')]")
    for a in anchors:
        link = a.get_attribute("href")
        if not link or link in seen:
            continue
        name = (a.get_attribute("aria-label") or a.text or "").strip()
        if not name:
            # Fallback: try inner name element
            try:
                name_el = a.find_element(By.XPATH, ".//div[contains(@class,'fontHeadlineSmall') or contains(@class,'qBF1Pd')]")
                name = name_el.text.strip()
            except Exception:
                name = ""
        if name:
            results.append({
                "place name":name,
                "category" :categoty,
                "city":city,
                "link":link
            })
            seen.add(link)
    return results

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

#os.mkdir("cities/")

def main():
    cities = ["Marrakech","Agadir", "Casablanca", "Fes", "Tanger", "Essaouira",
              "Chefchaouen", "Ouarzazate" ,"Merzouga", "Rabat", "Meknes","Tetouan",
              "Al Hoceima", "Oujda", "Saidia", "Laayoune", "Dakhla"]

    categories = ["Hotel", "Restaurant", "Tourist Attraction"]
    for city in cities:
        for category in categories:
            driver=make_driver()
            driver.get(build_search_url(category,city))
            feed=get_results_feed(driver)
            scroll_results(driver,feed)
            places=collect_places_from_feed(feed,category,city)
            save_to_csv(places,f"src/Google Maps/data/cities/{city}{category}_test.csv")
            driver.quit()


if __name__ == "__main__":
    main()