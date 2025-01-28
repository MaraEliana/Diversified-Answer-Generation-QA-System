import re
import time
import json
import os
import logging
import requests
import unicodedata
from openai import OpenAI
from langdetect import detect
from bs4 import BeautifulSoup
from lxml import html
from dotenv import load_dotenv
from utils import *

# load environment variables from the .env file
load_dotenv()

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

def normalize_text(text):
    # This function will normalize the text (e.g., remove unnecessary spaces, special characters, etc.)
    return ' '.join(text.split())


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


def decide_question(title, paragraph):
    pass

def parse_page(url, logger):
    """
    Parse a webpage and extract structured content.

    :param url: URL of the page to parse.
    :param logger: Logger object for logging messages.
    :return: Dictionary containing the extracted data.
    """
    
    # Get the page content
    response = requests.get(url)
    response.encoding = response.apparent_encoding or 'utf-8'
    response.raise_for_status()
    
    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the title (from the <title> tag)
    title_text = normalize_text(soup.title.string.strip() if soup.title else "No title found")
    if title_text == "No title found":
        logger.warning("Title not found")

    # Extract the content div using lxml for precise XPath targeting
    tree = html.fromstring(response.content.decode('utf-8', errors='replace'))
    content_xpath = '/html/body/div[5]/article/div/div/div[1]/div[3]'
    content_div = tree.xpath(content_xpath)
    if not content_div:
        logger.warning("Content div not found")
    content_div = content_div[0]

    # Extract the question field (first paragraph in content div, using the provided XPath)
    first_paragraph_xpath = '/html/body/div[5]/article/div/div/div[1]/div[1]/p'
    first_paragraph = tree.xpath(first_paragraph_xpath)
    first_paragraph = normalize_text(first_paragraph[0].text_content().strip() if first_paragraph else "No introductory paragraph found.")
    # NEXT TO CHANGE
    # decision = decide_question(title_text, first_paragraph)
    # if decision == "title":
    #     logger.info("Title chosen as the question")
    #     question = title_text
    # else:
    #     logger.info("Paragraph chosen as the question")
    #     question = first_paragraph
    question = first_paragraph

    # Extract all section titles (h2, h3, strong)
    section_titles = content_div.xpath(".//h2 | .//h3")

    # Create the answer as a dictionary
    answer = {}

    current_title = title_text
    section_content = []

    # Iterate over all elements in content_div, including paragraphs, h2, h3, strong, etc.
    for element in content_div.iter():  # iter() iterates over all child elements of content_div
        # If the element is a section title (h2, h3, or strong), this marks the start of a new section
        if element.tag in ['h2', 'h3']:
            # If we're already collecting content for a section, store the content for the previous section
            if section_content:
                answer[current_title] = " ".join(section_content)
            # Set the new section title as the current title
            current_title = normalize_text(element.text_content().strip())
            # Reset the section content to start collecting content for the new section
            section_content = []
        # If the element is a paragraph or list, add its content to the current section's content
        elif element.tag in ['p', 'ul', 'ol']:
            section_content.append(normalize_text(element.text_content().strip()))

    # After the loop, make sure to store the last section's content
    if section_content:
        answer[current_title] = " ".join(section_content)

    # Iterate over the answer keys and remove key-value pairs which are not in English
    multilingual = False
    for key in list(answer.keys()):
        if detect(answer[key]) != "en":
            multilingual = True
            del answer[key]

    # Extract links from the content
    links = [a.get("href") for a in content_div.xpath(".//p//a[@href]")]
    
    # Extract tags and their URLs from the specified tags section
    tags_xpath = '/html/body/div[5]/article/div/div/div[1]/div[5]'
    tags_div = tree.xpath(tags_xpath)
    if tags_div:
        tags = [normalize_text(tag.text_content().strip()) for tag in tags_div[0].xpath(".//a[@rel='tag']")]
        tag_urls = [tag.get("href") for tag in tags_div[0].xpath(".//a[@rel='tag']")]
    else:
        tags, tag_urls = [], []

    # Return the extracted page data
    # change the multiple_languages field
    page_data = {
        "html": response.text,
        "url": url,
        "date": extract_date_from_url(url, logger).strftime('%Y-%m-%d'), 
        "question": question,
        "answer": answer,
        "multiple_languages": multilingual,
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
    keys_to_extract = ["url", "date", "question", "answer", "multiple_languages", "links", "tags", "tag_urls"]
    problematic_urls = []
    # save each page to a .json file
    for i, url in enumerate(urls_list):
        logger.info(f"Parsing page: {url}")
        page_data = parse_page(url, logger)
        res = dict(filter(lambda item: item[0] in keys_to_extract, page_data.items()))
        # set the url to be the unique identifier for the document
        custom_id = url
        with open(f"pages/page_{i}.json", "w", encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        try:
            logger.info(f"Saving data to index: {index_name}")
            client.index(index=index_name, body=page_data, id=custom_id)
        except Exception as e:
            logger.error(f"Error processing URL {i} with '{url}': {e}")
            problematic_urls.append(res)
    logger.info(f"Saving {len(problematic_urls)} external URLs to 'problematic_urls.json'.")
    with open("problematic_urls.json", "w", encoding='utf-8') as f:
        json.dump(problematic_urls, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved {len(urls_list)} pages to index '{index_name}'.")

if __name__ == "__main__":
    logger.info("Script execution started.")
    opensearch_user = os.getenv('OPENSEARCH_USER')
    opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
    # create an OpenSearch client
    logger.info("Creating OpenSearch client.")
    opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)

    # create an index for the list of urls if it doesn't exist
    url_index = "eur-lex-diversified-urls-askep"
    url_mapping = load_mapping("./mappings/urls_mapping.json")
    ### DELETE THIS
    opensearch_client.indices.delete(index=url_index)
    ###
    create_index(opensearch_client, url_index, url_mapping, logger)

    # extract the new URLs
    new_urls = extract_new_urls(opensearch_client, url_index, logger)

    if new_urls:
        # save the new URLs to the QA index in OpenSearch
        qa_index = "eur-lex-diversified-qa-askep"
        qa_mapping = load_mapping("./mappings/qa_mapping.json")
        ### DELETE THIS
        opensearch_client.indices.delete(index=qa_index)
        ###
        # create the QA index if it doesn't exist
        create_index(opensearch_client, qa_index, qa_mapping, logger)
        # parse the new URLs and save the structured data to the QA index
        save_page_data(opensearch_client, qa_index, new_urls, logger)
    else:
        logger.info("No new URLs found. Exiting script.")
    logger.info("Script execution finished.")