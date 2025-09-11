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
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(2)  # petit implicite pour réduire le code
    return driver

def scroll_reviews(driver, scrolls=25, pause=1.3):
    """Scroll inside the reviews panel to load more reviews."""
    panel = driver.find_element(By.XPATH, "//div[contains(@class,'m6QErb DxyBCb kA9KIf dS8AEf XiKgde')]")
    last_height = 0
    for _ in range(scrolls):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", panel)
        time.sleep(pause)
        new_height = driver.execute_script("return arguments[0].scrollHeight", panel)
        if new_height == last_height:
            break
        last_height = new_height

def extract_rating(card):
    """Retourne la note trouvée, sinon '' """
    try:
        elem = card.find_element(By.XPATH, ".//span[contains(@class,'kvMYJc') or contains(@class,'fontBodyLarge fzvQIb')]")
        rating = elem.get_attribute("aria-label") or elem.text
        return rating.strip()
    except:
        return ""
    
def get_reviews(driver, place_url, max_reviews=50):
    reviews = []
    
    driver.get(place_url)
    wait = WebDriverWait(driver, 10)

    try:
        # Click on the "Reviews" button
        reviews_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label,'Avis') or contains(@aria-label,'Reviews')]")))
        reviews_btn.click()

        # Scroll to load reviews
        scroll_reviews(driver)

        # Collect review cards
        cards = driver.find_elements(By.XPATH, "//div[contains(@class,'jftiEf')]")
        if len(cards)>50:
            max_reviews=50
        else:
            max_reviews=len(cards)
        for card in cards[:max_reviews]:
            try:
                name = card.find_element(By.XPATH, ".//div[@class='d4r55 fontTitleMedium']").text
            except:
                name = ""
            rating = extract_rating(card)
            try:
                date = card.find_element(By.XPATH, ".//span[contains(@class,'xRkPPb') or contains(@class, 'rsqaWe')]").text
            except:
                date = ""

            try:
                text = card.find_element(By.XPATH, ".//span[contains(@class,'wiI7pd')]").text
            except:
                text = ""

            reviews.append({"name": name, "note": rating, "date": date,"date modifié": str(date).split("sur")[0] ,"review text": text})
            
    except Exception as e:
        print(f"[WARN] Could not scrape {place_url} – {e}")
    finally:
        driver.quit()
    return reviews

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def main():
    df=pd.read_csv("src/Google Maps/data/all_places.csv")
    reviews=[]
    for i in range(len(df)):
        url=df.iloc[i]["link"]
        print(f"scraping {df.iloc[i]['category']} of {df.iloc[i]['place name']} from {df.iloc[i]['city']}")
        driver=make_driver()
        results=get_reviews(driver, url)
        results=pd.DataFrame(results)
        for j in range(len(results)):
            reviews.append({
                "city":df.iloc[i]["city"],
                "category":df.iloc[i]["category"],
                "place name":df.iloc[i]["place name"],
                "name": results.iloc[j]["name"],
                "date" : results.iloc[j]["date modifié"],
                "review text":results.iloc[j]["review text"]
            })
    
    save_to_csv(reviews,"src/Google Maps/data/reviexs_test.csv")


if __name__ == "__main__":
    main()