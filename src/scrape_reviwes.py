from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from itertools import zip_longest
import time
import pandas as pd

def create_driver():
    chrome_options = Options()
    chrome_options.add_argument("--lang=en-US")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def is_blocked_page(driver):
    page_text = driver.page_source.lower()
    keywords = [
        "access denied", "verify you are human", "bot detected",
        "forbidden", "unusual traffic", "captcha", "blocked","Access blocked"
    ]
    return any(keyword in page_text for keyword in keywords)

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def scrape_reviews(df):
    results = []
    driver = None
    
    try:
        for i in range(len(df)):
            if i % 5 == 0:
                if driver:
                    driver.quit()
                    print("[INFO] Driver restarted after 5 pages")
                driver = create_driver()
                time.sleep(2)

            link = df.loc[i, "link"]
            print(f"\n[INFO] Scraping: {link}")
            try:
                driver.get(link)
                time.sleep(2)
                # Vérification anti-bot
                if is_blocked_page(driver):
                    print("[ALERT] Site a détecté un bot — fermeture du driver")
                    driver.quit()
                    driver = None
                    time.sleep(5)  # Attendre un peu avant de relancer
                    continue  # Passer au lien suivant

                selectors_to_try = [
                    '//div[@data-test-target="HR_CC_CARD"]',
                    '//div[@data-reviewid]',
                    '//div[contains(@data-test-target, "review")]',
                    '//section[@data-automation="reviews"]//div',
                    '//*[contains(@class, "review")]'
                ]

                element_found = False
                for selector in selectors_to_try:
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                        print(f"[OK] Found reviews with selector: {selector}")
                        element_found = True
                        break
                    except TimeoutException:
                        continue

                if not element_found:
                    print("[INFO] Trying fallback scroll + wait...")
                    time.sleep(5)
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(3)

                    try:
                        WebDriverWait(driver, 10).until(
                            lambda d: len(d.find_elements(By.XPATH, "//*[contains(text(), 'review') or contains(text(), 'Review')]")) > 0
                        )
                        print("[OK] Fallback review content detected")
                    except TimeoutException:
                        print("[WARN] Fallback also failed")

                titles = driver.find_elements(By.XPATH, '//div[contains(@data-test-target, "review-title")]')
                texts = driver.find_elements(By.XPATH, '//span[contains(@data-automation,"reviewText")]')

                for title, text in zip_longest(titles, texts, fillvalue=None):
                    results.append({
                        "review title": title.text if title else None,
                        "review": text.text if text else None,
                        "place name": df.loc[i, "place name"],
                        "category": df.loc[i, "category"],
                        "city": df.loc[i, "city"]
                    })

            except Exception as e:
                print(f"[ERROR] Failed to scrape {link}: {str(e)}")

    finally:
        if driver:
            driver.quit()
            print("[INFO] Final driver closed")

    return results

if __name__ == '__main__':
    df=pd.read_csv("src/all_places.csv")
    results=scrape_reviews(df)
    save_to_csv(results,"data/TripAdvisor_reviews_test.csv")