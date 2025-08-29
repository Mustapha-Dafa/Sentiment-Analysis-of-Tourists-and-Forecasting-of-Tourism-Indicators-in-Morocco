from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from itertools import zip_longest
import time,random , os
import pandas as pd
from urllib.parse import urljoin

def build_search_url(city: str) -> str:
    # Dates factices -> souvent plus stable. Adapte si tu veux.
    return (
        "https://www.booking.com/searchresults.html"
        f"?ss={city}&lang=fr&checkin=2025-11-10&checkout=2025-11-12&group_adults=2"
    )

MAX_PROPS_PER_CITY = 30      
DATA_DIR           = "data"

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

def sleep(a=2.0, b=4.0):
    time.sleep(random.uniform(a, b))

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def discover_properties_for_city(driver, city, limit=MAX_PROPS_PER_CITY):
    url = build_search_url(city)
    driver.get(url)
    sleep()
    if city=="Laayoune":
        limit=28
    props, seen = [], set()
    while len(props) < limit:
        # cartes d'établissement
        cards = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="property-card"]')
        for c in cards:
            try:
                a = c.find_element(By.CSS_SELECTOR, 'a[data-testid="title-link"]')
                href = a.get_attribute("href") or ""
                name = a.text.strip()
                full = urljoin("https://www.booking.com", href.split("?")[0])
                #print(f"name: {name} link: {full}")
                if full and full not in seen:
                    seen.add(full)
                    props.append({"city": city, "name": name, "url": full})
                    if len(props) >= limit:
                        break
            except:
                continue
        if len(props) >= limit:
            break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(0.5, 1.0)
        #next_btn.click()
        #sleep()
    driver.quit()
    return props

CITIES = ["Marrakech","Agadir", "Casablanca", "Fes", "Tanger", "Essaouira",
              "Chefchaouen", "Ouarzazate" ,"Merzouga", "Rabat", "Meknes","Tetouan",
              "Al Hoceima", "Oujda", "Saidia", "Laayoune", "Dakhla"]

for city in CITIES:
    driver = make_driver()
    print(f"scraping {city}")
    L=discover_properties_for_city(driver,city)
    save_to_csv(L,"src/Booking/data/"f"{city}_test.csv")
