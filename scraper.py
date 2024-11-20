import re
import time
import json
import os
import logging
import requests
from bs4 import BeautifulSoup
from lxml import html
from dotenv import load_dotenv
from utils import *

# setup logging
script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(script_dir, 'scraper.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# base URL
BASE_URL = "https://epthinktank.eu/author/epanswers/page/{}"

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

def extract_new_urls(client, index_name, logger=None):
    """
    :param client: OpenSearch client.
    :param index_name: Name of the index.
    :param logger: Optional logger object.
    :return: List of new URLs.
    """
    # get the latest date from the index
    latest_date = get_latest_date(client, index_name, logger)
    if latest_date:
        # get the URLs corresponding to the latest date
        latest_urls = get_urls_by_date(client, index_name, latest_date, logger)
    else:
        logger.info("No latest date found in the index.")
        latest_urls = []
        
    # scraping pages
    page = 1
    stop_scraping = False
    new_urls = []
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
                # add the new url to the list of new urls
                new_urls.append(url)
    
        page += 1
        time.sleep(1)
    return new_urls


def parse_page(url, logger):
    """
    Parse a webpage and extract structured content.

    :param url: URL of the page to parse.
    :param logger: Logger object for logging messages.
    :return: Dictionary containing the extracted data.
    """

    response = requests.get(url)
    response.raise_for_status()
    tree = html.fromstring(response.content)
    
    # extract date
    date = extract_date_from_url(url, logger)

    # extract title
    title_xpath = '/html/body/div[5]/article/header/div[2]/div/div/div/h1'
    title = tree.xpath(title_xpath)
    title_text = title[0].text_content().strip() if title else "No title found"
    if title_text == "No title found":
        logger.warning("Title not found")

    # target the content div
    content_xpath = '/html/body/div[5]/article/div/div/div[1]/div[3]'
    content_div = tree.xpath(content_xpath)
    if not content_div:
        logger.warning("Content div not found")
    content_div = content_div[0]

    # extract the question field (first paragraph in content_div)
    first_paragraph = content_div.xpath(".//p[1]")
    question = first_paragraph[0].text_content().strip() if first_paragraph else "No introductory paragraph found."
    if question == "No introductory paragraph found.":
        logger.warning("Question not found")

    # extract all section titles (h2, h3, strong)
    section_titles = content_div.xpath(".//h2 | .//h3 | .//strong")

    # create the answer as a dictionary
    answer = {}
    
    # iterate through the section titles
    for i, section_title in enumerate(section_titles):
        title_text = section_title.text_content().strip()
        
        # get the start of the current section
        start_element = section_title.getnext()
        
        # determine the end of the current section
        if i + 1 < len(section_titles):
            # there is a next section title
            end_element = section_titles[i + 1]
        else:
            # this is the last section, include everything until the end of the div
            end_element = None

        # collect all content between the current and next title
        section_content = []
        while start_element is not None and start_element != end_element:
            if start_element.tag in ['p', 'ul', 'ol']:
                section_content.append(start_element.text_content().strip())
            start_element = start_element.getnext()

        # join the section content and store it
        answer[title_text] = " ".join(section_content)

    # extract links from the content
    links = [a.get("href") for a in content_div.xpath(".//p//a[@href]")]
    
    # extract tags and their URLs from the specified tags section
    tags_xpath = '/html/body/div[5]/article/div/div/div[1]/div[5]'
    tags_div = tree.xpath(tags_xpath)
    if tags_div:
        tags = [tag.text_content().strip() for tag in tags_div[0].xpath(".//a[@rel='tag']")]
        tag_urls = [tag.get("href") for tag in tags_div[0].xpath(".//a[@rel='tag']")]
    else:
        tags, tag_urls = [], []

    page_data = {
        "html": response.text,
        "url": url,
        "date": date,
        "title": title_text,
        "question": question,
        "answer": answer,
        "links": links,
        "tags": tags,
        "tag_urls": tag_urls
    }

    return page_data

def save_page_data(client, index_name, urls_list, logger=None):
    """
    Parse the content of the URLs in the list and save the structured data to the QA index.
    :param client: OpenSearch client.
    :param index_name: Name of the index.
    :param urls_list: List of URLs to parse and save.
    :param logger: Optional logger object.
    """
    unindexed_pages = []
    for i, url in enumerate(urls_list):
        try:
            logger.info(f"Parsing page: {url}")
            page_data = parse_page(url, logger)
            logger.info(f"Saving data to index: {index_name}")
            client.index(index=index_name, body=page_data)
        except Exception as e:
            logger.error(f"Error processing URL '{url}': {e}")
            unindexed_pages.append(page_data)
    # buggy at the moment, unaccepted date type
    # with open("unindexed_pages.json", "w") as f:
    #     json.dump(unindexed_pages, f)
    logger.info(f"Saved {len(urls_list)} pages to index '{index_name}'.")

if __name__ == "__main__":
    logger.info("Script execution started.")
    # load environment variables from the .env file
    load_dotenv()
    opensearch_user = os.getenv('OPENSEARCH_USER')
    opensearch_password = os.getenv('OPENSEARCH_PASSWORD')

    # create an OpenSearch client
    logger.info("Creating OpenSearch client.")
    opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)

    # create an index for the list of urls if it doesn't exist
    url_index = "eur-lex-diversified-urls-askep"
    ##### DELETE INDEX
    # opensearch_client.indices.delete(index=url_index, ignore=[400, 404])
    ##### 
    url_mapping = load_mapping("./mappings/urls_mapping.json")
    create_index(opensearch_client, url_index, url_mapping, logger)

    # extract the new URLs
    new_urls = extract_new_urls(opensearch_client, url_index, logger)
    if new_urls:
        # save the new URLs to the QA index in OpenSearch
        qa_index = "eur-lex-diversified-qa-askep"
        ##### DELETE INDEX
        # opensearch_client.indices.delete(index=qa_index, ignore=[400, 404])
        #####
        qa_mapping = load_mapping("./mappings/qa_mapping.json")
        # create the QA index if it doesn't exist
        create_index(opensearch_client, qa_index, qa_mapping, logger)

        # parse the new URLs and save the structured data to the QA index
        save_page_data(opensearch_client, qa_index, new_urls, logger)
    else:
        logger.info("No new URLs found. Exiting script.")
    logger.info("Script execution finished.")