import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
import re
import time
import json
import os
from datetime import datetime
import logging

# setup logging
script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(script_dir, 'url_scraper.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# base URL
BASE_URL = "https://epthinktank.eu/author/epanswers/page/{}"
URL_FILE = os.path.join(script_dir, 'scraped_urls.json')

# regular expression pattern to match URLs with the required date format
url_pattern = re.compile(r"^https://epthinktank\.eu/\d{4}/\d{2}/\d{2}/")

def load_existing_urls():
    # Check if the file exists and load if non-empty
    if os.path.exists(URL_FILE) and os.path.getsize(URL_FILE) > 0:
        logger.info(f"Loading existing URLs from {URL_FILE}")
        with open(URL_FILE, 'r') as file:
            data = json.load(file)
            return data.get("urls", [])
    logger.info("No existing URLs found. Starting fresh.")
    return []

def save_urls(url_list, new_url_count):
    logger.info(f"Saving URLs to {URL_FILE}. New URLs: {new_url_count}")
    data = {
        "urls": url_list,
        "new_url_count": new_url_count
    }
    with open(URL_FILE, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    logger.info(f"URLs saved successfully to {URL_FILE}")

def scrape_blog_urls(page_number):
    url = BASE_URL.format(page_number)
    logger.info(f"Requesting page {page_number} from {url}")
    response = requests.get(url)
    
    # check if the page exists
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all(href=re.compile(url_pattern))
        if not links:
            logger.info(f"No articles found on page {page_number}. Ending scraping.")
            return False
        
        for link in links:
            url = link['href']
            new_urls.append(url)
        
        logger.info(f"Found {len(new_urls)} new URLs on page {page_number}")
        return new_urls
    else:
        logger.warning(f"Page {page_number} not found (status code {response.status_code}). Ending scraping.")
        return False

if __name__ == "__main__":
    logger.info("Script execution started.")

    # load previously scraped URLs
    existing_urls = load_existing_urls()

    # if there are existing URLs, get the first one, which should be the most recent
    if existing_urls:
        last_scraped_url = existing_urls[0]  # the first entry is the most recent URL
        logger.info(f"Starting from the most recent URL: {last_scraped_url}")
    else:
        last_scraped_url = None  # if no previous data, start scraping from the first page
        logger.info("No last scraped URL. Starting fresh.")

    # store new URLs while preserving insertion order
    new_urls = []

    # scraping pages
    page = 1
    while True:
        logger.info(f"Scraping URLs from page {page}")
        
        # scrape URLs from the current page
        scraped_urls = scrape_blog_urls(page)
        
        # if no URLs found or invalid page, break the loop
        if not scraped_urls:
            logger.info("Stopping scraping as no new URLs were found or end of pages reached.")
            break
        
        # add new URLs to the list
        new_urls.extend(scraped_urls)
        
        # stop if we encounter the last scraped URL
        if last_scraped_url and last_scraped_url in scraped_urls:
            logger.info("Found the most recent article URL. Stopping scraping.")
            break
        
        page += 1
        time.sleep(1)  # delay to avoid overloading the server

    # combine old URLs with newly scraped ones, ensuring uniqueness
    all_urls = list(OrderedDict.fromkeys(new_urls + existing_urls)) # preserves order, newer urls will be at the beginning
    new_urls = list(OrderedDict.fromkeys(new_urls))
    new_url_count = len(new_urls)

    # save the updated URLs and the count of new URLs to the file
    save_urls(all_urls, new_url_count)
    logger.info(f"Collected newest URLs. Everything up to date. Total URLs: {len(all_urls)}. New URLs: {new_url_count}")
    logger.info("Script execution finished.")