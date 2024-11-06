import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
import re
import time
import json
import os

# base URL for pagination
BASE_URL = "https://epthinktank.eu/author/epanswers/page/{}"
URL_FILE = 'scraped_urls.json'  # File to store the list of URLs

# regular expression pattern to match URLs with the required date format
url_pattern = re.compile(r"^https://epthinktank\.eu/\d{4}/\d{2}/\d{2}/")

# load the existing URLs from the file if it exists
def load_existing_urls():
    if os.path.exists(URL_FILE) and os.path.getsize(URL_FILE) > 0:
        with open(URL_FILE, 'r') as file:
            return json.load(file)
    return []

# save the list of URLs to the file
def save_urls(url_list):
    with open(URL_FILE, 'w') as file:
        json.dump(url_list, file, indent=4)

# scrape blog post URLs from a single page
def scrape_blog_urls(page_number):
    url = BASE_URL.format(page_number)
    response = requests.get(url)
    
    # check if the page exists
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # locate all div elements inside the specific container structure
        article_divs = soup.select("body > div.c-black.f-text.main > div.container.section-latest.mt-3.mb-5.pb-5.section > div.row.infinite-scroll > div")
        
        # stop if no articles were found on the page
        if not article_divs:
            return False
        
        # loop through each article div and search for nested <a> tags
        new_urls = []
        for article in article_divs:
            links = article.find_all("a", href=True)
            for link in links:
                url = link['href']
                
                # check if the URL matches the pattern
                if url_pattern.match(url):
                    new_urls.append(url)
        
        return new_urls
    else:
        # return False if page does not exist (e.g., 404 error)
        return False

if __name__ == "__main__":
    # load previously scraped URLs
    existing_urls = load_existing_urls()

    # if there are existing URLs, get the first one, which should be the most recent
    if existing_urls:
        last_scraped_url = existing_urls[0]  # The first entry is the most recent URL
        print(f"Starting from the most recent URL: {last_scraped_url}")
    else:
        last_scraped_url = None  # If no previous data, start scraping from the first page

    # set to store new URLs while preserving insertion order
    new_urls = []

    # scraping pages
    page = 1
    while True:
        print(f"Scraping URLs from page {page}")
        
        # scrape URLs from the current page
        scraped_urls = scrape_blog_urls(page)
        
        # if no URLs found or invalid page, break the loop
        if not scraped_urls:
            print("No more articles found or end of pages reached.")
            break
        
        # add new URLs to the list
        new_urls.extend(scraped_urls)
        
        # stop if we encounter the last scraped URL
        if last_scraped_url and last_scraped_url in scraped_urls:
            print("Found the most recent article URL. Stopping scraping.")
            break
        
        page += 1
        time.sleep(1)  # Delay to avoid overloading the server

    # combine old URLs with newly scraped ones, ensuring uniqueness
    all_urls = list(OrderedDict.fromkeys(new_urls + existing_urls))

    # save the updated URLs to the file
    save_urls(all_urls)
    print(f"Collected newest URLs. Everything up to date. Total URLs: {len(all_urls)}")
