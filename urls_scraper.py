import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
import re
import time
import json
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
from opensearch_utils import *

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
        new_urls = []
        for link in links:
            url = link['href']
            new_urls.append(url)
        
        # Remove duplicates while preserving order
        new_urls = list(dict.fromkeys(new_urls))
        
        logger.info(f"Found {len(new_urls)} new URLs on page {page_number}")
        return new_urls
    else:
        logger.warning(f"Page {page_number} not found (status code {response.status_code}). Ending scraping.")
        return False

if __name__ == "__main__":
    logger.info("Script execution started.")
    # load environment variables from the .env file
    load_dotenv()
    opensearch_user = os.getenv('OPENSEARCH_USER')
    opensearch_password = os.getenv('OPENSEARCH_PASSWORD')

    # create an OpenSearch client
    logger.info("Creating OpenSearch client.")
    opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)
    # create an index if it doesn't exist
    index_name = "eur-lex-diversified-urls-askep"
    # delete this
    # response = opensearch_client.indices.delete(index=index_name, ignore=[400, 404])
    mapping = {
                "mappings": {
                    "properties": {
                        "url": {"type": "keyword"},
                        "date": {"type": "date"}
                    }
                }
            }
    create_index(opensearch_client, index_name, mapping, logger)

    # get the latest date from the index
    latest_date = get_latest_date(opensearch_client, index_name, logger)
    if latest_date:
        # get the URLs corresponding to the latest date
        latest_urls = get_urls_by_date(opensearch_client, index_name, latest_date, logger)
    else:
        logger.info("No latest date found in the index.")
        latest_urls = []
        
    # scraping pages
    page = 1
    stop_scraping = False

    while not stop_scraping:
        logger.info(f"Scraping URLs from page {page}")
        
        # scrape URLs from the current page
        scraped_urls = scrape_blog_urls(page)
        
        # if no URLs found or invalid page, break the loop
        if not scraped_urls:
            logger.info("Stopping scraping as no new URLs were found or end of pages reached.")
            break
        
        for url in scraped_urls:
            if latest_urls and url in latest_urls:
                logger.info("Found the most recent article URL. Stopping scraping.")
                stop_scraping = True
                break
            else:
                # add the new url to the opensearch index
                logger.info(f"Adding URL '{url}' to the index.")
                add_url_with_date(opensearch_client, index_name, url, logger)
    
        page += 1
        time.sleep(1)
    logger.info("Script execution finished.")