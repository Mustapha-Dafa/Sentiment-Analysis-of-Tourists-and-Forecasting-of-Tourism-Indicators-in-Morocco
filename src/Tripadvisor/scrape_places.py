from selenium import webdriver
import time 
import re
import random
import requests
import logging
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import (
    NoSuchElementException, 
    ElementNotInteractableException,
    TimeoutException,
    ElementClickInterceptedException
)
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd

chrome_options = Options()
chrome_options.add_argument("--lang=en-US")
#chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu") 
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)
service = Service()

def scrape(url):
    driver = webdriver.Chrome(options=chrome_options,service=service)
    driver.get(url)
    
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((
            By.XPATH, 
            '//*[contains(@data-test-attribute, "all-results-section")]'
        ))
    )
    
    try:
        driver.find_element(
            By.XPATH,
            '//button[contains(text(), "Accept")]'
        ).click()
        time.sleep(2)
    except NoSuchElementException:
        pass
    
    # Find and click "Show more" button with multiple strategies
        show_more_clicked = False
        
        # Strategy 1: Wait for element to be clickable
        try:
            show_more_button = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    '//button//*[contains(text(), "Show more")]/ancestor::button'
                ))
            )
            show_more_button.click()
            show_more_clicked = True
            print("Show more button clicked successfully")
        except (TimeoutException, ElementNotInteractableException, ElementClickInterceptedException):
            pass
        
        # Strategy 2: Try alternative XPath
        if not show_more_clicked:
            try:
                show_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((
                        By.XPATH,
                        '//button[contains(text(), "Show more")]'
                    ))
                )
                show_more_button.click()
                show_more_clicked = True
                print("Show more button clicked with alternative XPath")
            except (TimeoutException, ElementNotInteractableException, ElementClickInterceptedException):
                pass
        
        # Strategy 3: Scroll to element and use ActionChains
        if not show_more_clicked:
            try:
                show_more_elements = driver.find_elements(
                    By.XPATH,
                    '//*[contains(text(), "Show more")]'
                )
                
                for element in show_more_elements:
                    try:
                        # Scroll to element
                        driver.execute_script("arguments[0].scrollIntoView(true);", element)
                        time.sleep(1)
                        
                        # Try clicking with ActionChains
                        actions = ActionChains(driver)
                        actions.move_to_element(element).click().perform()
                        show_more_clicked = True
                        print("Show more button clicked with ActionChains")
                        break
                    except Exception as e:
                        continue
            except Exception as e:
                print(f"ActionChains strategy failed: {e}")
        
        # Strategy 4: JavaScript click as last resort
        if not show_more_clicked:
            try:
                show_more_elements = driver.find_elements(
                    By.XPATH,
                    '//*[contains(text(), "Show more")]'
                )
                
                for element in show_more_elements:
                    try:
                        driver.execute_script("arguments[0].click();", element)
                        show_more_clicked = True
                        print("Show more button clicked with JavaScript")
                        break
                    except Exception as e:
                        continue
            except Exception as e:
                print(f"JavaScript click failed: {e}")
        
        if not show_more_clicked:
            print("Warning: Could not click 'Show more' button")
        
        # Wait for additional content to load after clicking
        time.sleep(5)
    page_source = driver.page_source
    driver.quit()
    return page_source

def parse(html):
    soup = BeautifulSoup(html, 'html.parser')
    listings = []
    
    for listing in soup.select('[data-test-attribute="location-results-card"]'):
        title = listing.select_one('.FGwzt')
        href = listing.select_one('a').get('href')

        listings.append({
            'place name': title.text,
            'link': 'https://www.tripadvisor.com' + href
        })
    
    return listings

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

cities = ["Marrakech","Agadir", "Casablanca", "Fes", "Tanger", "Essaouira",
              "Chefchaouen", "Ouarzazate" ,"Merzouga", "Rabat", "Meknes","Tetouan",
              "Al Hoceima", "Oujda", "Saidia", "Laayoune", "Dakhla"]

categories = ["Hotel", "Restaurant", "Tourist Attraction"]

if __name__ == '__main__':
    places = []
    for city in cities:
        for cat in categories:
            print(f"scraping {city} city")
            cat=cat.replace(' ','+')
            query =f'{cat}+in+{city}'
            url=f'https://www.tripadvisor.com/Search?q={query}'
            html=scrape(url)
            results=parse(html)
            for i in range (len(results)):
                places.append({
                    'place name':results[i]['place name'],
                    'category' : cat,
                    'city' : city,
                    'link' : results[i]['link']
                })
    save_to_csv(places ,"all_places.csv")