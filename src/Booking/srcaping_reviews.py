from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from itertools import zip_longest
import time,random , os
import pandas as pd
from urllib.parse import urljoin

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

def open_reviews_tab(driver):
    # essaie plusieurs libellés/boutons pour atteindre l’onglet Avis
    for sel in [
        "a[href*='tab=reviews']",
        "button:contains('Commentaires')",  # ne marche pas en CSS natif mais on laisse l’idée
    ]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            el.click(); sleep(2,3); return True
        except:
            pass
    # fallback: clique des boutons par texte (XPath)
    for xp in [
        "//a[contains(@href, 'tab=reviews')]",
        "//button[contains(., 'Commentaires')]",
        "//button[contains(., 'Avis')]",
        "//button[contains(., 'Reviews')]",
        "//a[contains(., 'Commentaires')]",
        "//a[contains(., 'Avis')]",
        "//a[contains(., 'Reviews')]",
    ]:
        try:
            el = driver.find_element(By.XPATH, xp)
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            sleep(0.3, 0.8)
            el.click()
            sleep(2, 3)
            return True
        except:
            continue
    return False

df=pd.read_csv("src/Booking/data/all_places.csv")
cities = ["Dakhla"]
results=[]
for city in cities:
    driver = make_driver()
    print(f"scraping {city} city")
    for i in range (len(df[df["city"]==city])):
        url=df[df["city"]==city].iloc[i]["url"]
        driver.get(url)
        sleep()
        if open_reviews_tab(driver):
            print("la page est trouvé")

            page_source = driver.page_source
        else:
            print("la page n'est pas trouvé")
        soup = BeautifulSoup(page_source, 'html.parser')
        if soup.select('[data-testid="review-card"]'):
            for listing in soup.select('[data-testid="review-card"]'):
                auther = listing.select_one('.f546354b44')
                note = listing.select_one('.bc946a29db')
                natoin = listing.select_one('.fb14de7f14')
                date = listing.select_one('[data-testid="review-date"]')
                title = listing.select_one('[data-testid="review-title"]')
                positive_text = listing.select_one('[data-testid="review-positive-text"]')
                negative_text = listing.select_one('[data-testid="review-negative-text"]')
                

                try :
                    title= title.text
                except : 
                    title=''
                try :
                    positive_text=positive_text.text
                except : 
                    positive_text=''
                try :
                    negative_text=negative_text.text
                except : 
                    negative_text=''
                
                
                results.append({
                    'city':city,
                    'place name':df[df["city"]==city].iloc[i]["name"],
                    'name': auther.text,
                    'note': note.text.replace('Avec une note de',''),
                    'nationality': natoin.text,
                    'date': date.text.replace('Commentaire envoyé le','').replace('\xa0','/'),
                    'review title': title.replace('\xa0',''),
                    'positive_text':positive_text,
                    'negative_text' :negative_text
                })
        else :
            print("pas d'avis dans ce hébergement")
            continue
    driver.quit()

save_to_csv(results,"src/Booking/data/reviwes_test.csv")
