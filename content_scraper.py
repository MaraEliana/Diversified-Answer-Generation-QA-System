import json
import os
import requests
from bs4 import BeautifulSoup
from lxml import html
from datetime import datetime

# get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# combine it with the desired filename to get the full path
URL_FILE = os.path.join(script_dir, 'scraped_urls.json') # file to store the scraped URLs
# combine it with the desired filename to get the full path
CONTENT_FILE = os.path.join(script_dir, 'extracted_data.json') # file to store the scraped URLs

# current time as a string
def current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# extract date from URL
def extract_date_from_url(url):
    date_parts = url.split("https://epthinktank.eu/")[1].split("/")[0:3]
    return "-".join(date_parts)  # Converts "YYYY/MM/DD" to "YYYY-MM-DD" for storage in OpenSearch

# parse each page and extract information
def parse_page(url):
    date = extract_date_from_url(url)

    # send HTTP request to get the page content
    response = requests.get(url)
    response.raise_for_status()  # Check for request errors

    # parse HTML content with BeautifulSoup and lxml
    soup = BeautifulSoup(response.content, "html.parser")
    tree = html.fromstring(response.content)

    # extract title (assuming only one h1 in the article header)
    title = tree.xpath('/html/body/div[5]/article/header/div[2]/div/div/div/h1')
    title_text = title[0].text_content().strip() if title else "No title found"

    # Target the specific content div using XPath
    content_div = tree.xpath('/html/body/div[5]/article/div/div/div[1]/div[3]')
    if content_div:
        content_div = content_div[0]
        # Section titles (collecting h2, h3 tags within the specific div)
        section_titles = [heading.text_content().strip() for heading in content_div.xpath(".//h2 | .//h3")]

        # Paragraph texts within the specific content div
        section_texts = [p.text_content().strip() for p in content_div.xpath(".//p")]

        # Links within paragraphs in the content div
        links = [a.get("href") for a in content_div.xpath(".//p//a[@href]")]
    else:
        section_titles, section_texts, links = [], [], []

    # Extract tags and their URLs from the specified tags section
    tags_div = tree.xpath('/html/body/div[5]/article/div/div/div[1]/div[5]')
    if tags_div:
        tags = [tag.text_content().strip() for tag in tags_div[0].xpath(".//a[@rel='tag']")]
        tag_urls = [tag.get("href") for tag in tags_div[0].xpath(".//a[@rel='tag']")]
    else:
        tags, tag_urls = [], []
    
    page_data = {
        "url": url,
        "date": date,
        "title": title_text,
        "section_titles": section_titles,
        "section_texts": section_texts,
        "links": links,
        "tags": tags,
        "tag_urls": tag_urls
    }

    return page_data

# Load URLs from JSON file
def load_urls():
    print(f"[{current_time()}] Loading URLs from {URL_FILE}...")
    with open(URL_FILE, "r") as file:
        data = json.load(file)
    print(f"[{current_time()}] Loaded {len(data['urls'])} URLs.")
    return data["urls"], data["new_url_count"]

# Load the existing extracted data
def load_existing_extracted_data():
    print(f"[{current_time()}] Loading existing extracted data from {CONTENT_FILE}...")
    if os.path.exists(CONTENT_FILE) and os.path.getsize(CONTENT_FILE) > 0:
        with open(CONTENT_FILE, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
        print(f"[{current_time()}] Loaded {len(existing_data)} existing entries.")
        return existing_data
    else:
        print(f"[{current_time()}] No existing extracted data found.")
        return []

# Save the updated data to the extracted_data.json file
def save_extracted_data(all_data):
    print(f"[{current_time()}] Saving updated data to {CONTENT_FILE}...")
    with open(CONTENT_FILE, "w", encoding="utf-8") as output_file:
        json.dump(all_data, output_file, ensure_ascii=False, indent=4)
    print(f"[{current_time()}] Data saved. Total entries: {len(all_data)}.")

if __name__ == "__main__":
    print(f"[{current_time()}] Script execution started.")

    # Load URLs and new URL count from scraped_urls.json
    urls, new_url_count = load_urls()

    # Get the last new_url_count URLs
    last_urls = urls[-new_url_count:]

    # Load previously extracted data
    existing_data = load_existing_extracted_data()

    # Scrape the last pages
    new_data = []
    # TODO: DO NOT FORGET TO CHANGE THIS!!!
    for url in last_urls[0:5]: 
        print(f"[{current_time()}] Parsing {url}...")
        try:
            page_data = parse_page(url)
            new_data.append(page_data)
            print(f"[{current_time()}] Successfully parsed {url}")
        except Exception as e:
            print(f"[{current_time()}] Error parsing {url}: {e}")

    # Combine existing data with the newly scraped data
    all_data = existing_data + new_data

    # Save the updated data to extracted_data.json
    save_extracted_data(all_data)

    print(f"[{current_time()}] Data extraction complete. Updated 'extracted_data.json' with {len(new_data)} new entries.")