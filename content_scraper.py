import json
import os
import requests
from bs4 import BeautifulSoup
from lxml import html

# extract date from URL
def extract_date_from_url(url):
    date_parts = url.split("https://epthinktank.eu/")[1].split("/")[0:3]
    return "-".join(date_parts)  # Converts "YYYY/MM/DD" to "YYYY-MM-DD" for storage in OpenSearch

# parse each page and extract information
def parse_page(url):
    # extract date
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

    # target the specific content div using XPath
    content_div = tree.xpath('/html/body/div[5]/article/div/div/div[1]/div[3]')
    if content_div:
        content_div = content_div[0]

        # section titles (collecting h1, h2, h3 tags within the specific div)
        section_titles = [heading.text_content().strip() for heading in content_div.xpath(".//h1 | .//h2 | .//h3")]

        # paragraph texts within the specific content div
        section_texts = [p.text_content().strip() for p in content_div.xpath(".//p")]

        # links within paragraphs in the content div
        links = [a.get("href") for a in content_div.xpath(".//p//a[@href]")]
    else:
        section_titles, section_texts, links = [], [], []

    # extract tags and their URLs from the specified tags section
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

# load URLs from JSON file
def load_urls():
    with open("scraped_urls.json", "r") as file:
        data = json.load(file)
    return data["urls"], data["new_url_count"]

# load the existing extracted data
def load_existing_extracted_data():
    if os.path.exists("extracted_data.json") and os.path.getsize("extracted_data.json") > 0:
        with open("extracted_data.json", "r", encoding="utf-8") as file:
            return json.load(file)
    return []

# save the updated data to the extracted_data.json file
def save_extracted_data(all_data):
    with open("extracted_data.json", "w", encoding="utf-8") as output_file:
        json.dump(all_data, output_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # load URLs and new URL count from scraped_urls.json
    urls, new_url_count = load_urls()

    # get the last new_url_count URLs
    last_urls = urls[-new_url_count:]

    # load previously extracted data
    existing_data = load_existing_extracted_data()

    # scrape the last N pages
    new_data = []
    # DO NOT FORGET TO CHANGE THIS!!!
    for url in last_urls[0:5]:
        print(f"Parsing {url}")
        try:
            page_data = parse_page(url)
            new_data.append(page_data)
        except Exception as e:
            print(f"Error parsing {url}: {e}")

    # combine existing data with the newly scraped data
    all_data = existing_data + new_data

    # save the updated data to extracted_data.json
    save_extracted_data(all_data)

    print(f"Data extraction complete. Updated 'extracted_data.json' with {len(new_data)} new entries.")