from crawl4ai import AsyncWebCrawler
import asyncio
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, RecursiveCharacterTextSplitter
import tiktoken
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import logging

LOG_FILE = "index_pages.log"
client = OpenAI(
    api_key="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",  # This is the default and can be omitted
)
# initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def scrape_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract and return the raw text (without tags)
        raw_text = soup.get_text(separator="\n")  # Use newline separator for better readability
        return raw_text.strip()  # Remove leading/trailing whitespace
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error scraping URL {url}: {e}")
        return None

async def scrape_text_with_crawl4ai(url):
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
        return result.markdown
    
def split_text_with_char_splitter(text, max_chars=6000):
    # Initialize the text splitter with a character chunk size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,  # Max number of characters per chunk
        chunk_overlap=200,  # Number of characters to overlap between chunks
        length_function=len
    )
    return splitter.split_text(text)

def embed_text_chunks(chunks, model="text-embedding-3-small"):
    embeddings = []
    for chunk in chunks:
        embeddings.append(client.embeddings.create(input = [chunk], model=model).data[0].embedding)
    return embeddings

# Initialize tokenizer for OpenAI's model (cl100k_base)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Function to count tokens in a text
def count_tokens(text):
    return len(tokenizer.encode(text))

# read URLs from the JSON file
def read_urls_from_file(file_path):
    logger.info(f"Reading URLs from file: {file_path}")
    try:
        with open(file_path, 'r') as file:
            urls = json.load(file)
            logger.info(f"Loaded {len(urls)} URLs from file")
            return urls
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading URLs from file: {e}")
        return []

async def main():
    URLS_FILE = "non_pdf_urls.json"
    urls = read_urls_from_file(URLS_FILE)
    urls = urls[:5]
    for url in tqdm(urls):
            logger.info(f"Processing URL: {url}")
            # basic
            # text = scrape_text(url)
            # advanced
            text = await scrape_text_with_crawl4ai(url)
            logger.info(f"Raw text from url {url}: {text}\n\n")
            if text:
                chunks = split_text_with_char_splitter(text)
                logger.info(f"Number of chunks: {len(chunks)}\n")
                for i, chunk in enumerate(chunks):
                    logger.info(f"Token number of chunk {i+1}: {count_tokens(chunk)}")
                    logger.info(f"Chunk {i+1}: {chunk}\n")
                embeddings = embed_text_chunks(chunks)
                logger.info(f"Number of embeddings: {len(embeddings)}\n")

if __name__ == "__main__":
    asyncio.run(main())