import os
import logging
import requests
import json
import time
import ollama
import tiktoken
from openai import OpenAI, APIError
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch
from utils import create_opensearch_client, create_index
from crawl4ai import AsyncWebCrawler
import asyncio

# config files
URLS_FILE = "non_pdf_urls.json"
LOG_FILE = "non_pdfs.log"

# constants
TOKEN_LIMIT_PER_MINUTE = 1_000_000  # 1 million tokens per minute
SECONDS_IN_MINUTE = 60

# load environment variables from the .env file
load_dotenv()
opensearch_user = os.getenv('OPENSEARCH_USER')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
    """
    Scrapes the text content from a URL.
    :param url: URL to scrape.
    :return: Cleaned text content.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        
        # extract raw text
        raw_text = soup.get_text(separator="\n")
        
        # remove empty lines and strip extra whitespace
        clean_text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())
        
        return clean_text
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error scraping URL {url}: {e}")
        return None
    
async def scrape_text_with_crawl4ai(url):
    """
    Scrapes the text content from a URL using Crawl4AI.
    :param url: URL to scrape.
    :return: Cleaned text content.
    """
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

def split_text_with_char_splitter(text, max_chars=6000):
    """
    Splits the text into chunks using RecursiveCharacterTextSplitter.
    :param text: Text content to split.
    :param max_chars: Maximum number of characters per chunk.
    :return: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=200, 
        length_function=len
    )
    return splitter.split_text(text)

def calculate_tokens_for_chunks(chunks):
    """
    Calculates the total number of tokens required to encode the text chunks.
    :param chunks: List of text chunks.
    :return: Total number of tokens required.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    for chunk in chunks:
        # encode the chunk to get the tokens and count them
        total_tokens += len(enc.encode(chunk))
    
    return total_tokens

def embed_text_with_ollama(embedding_model, chunks):
    """
    Embeds the text chunks using the Ollama model.
    :param embedding_model: Ollama embedding model to use.
    :param chunks: List of text chunks to embed.
    :return: List of embeddings corresponding to the text chunks.
    """
    logger.info(f"Embedding {len(chunks)} chunks with Ollama model: {embedding_model}")
    embeddings = []
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(model=embedding_model, prompt=chunk)
        embedding = response['embedding']
        if i == 0:
            logger.info(f"Embedding dimensions: {len(embedding)}")
        if embedding:
            embeddings.append(embedding)
            logger.debug(f"Successfully embedded chunk {i}")
        else:
            embeddings.append(None)
    return embeddings

def embed_text_with_openai(chunks, model="text-embedding-3-small"):
    """
    Embeds the text chunks using the OpenAI model.
    :param chunks: List of text chunks to embed.
    :param model: OpenAI embedding model to use.
    :return: List of embeddings corresponding to the text chunks.
    """
    logger.info(f"Embedding {len(chunks)} chunks with OpenAI's embedding model: {model}")
    embeddings = []
    for chunk in chunks:
        embeddings.append(client.embeddings.create(input = [chunk], model=model).data[0].embedding)
    return embeddings


def index_embeddings(index_name, chunks, embeddings, url, opensearch_client, logger):
    """
    Embeds and indexes the text chunks in OpenSearch.
    :param index_name: Index name.
    :param chunks: List of text chunks.
    :param embeddings: List of embeddings corresponding to the text chunks.
    :param url: URL of the page.
    :return: List of unindexed chunks with corresponding chunk_id and url.
    """
    has_unindexed_chunks = False
    logger.info(f"Indexing embeddings for URL: {url}.")
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if embedding:
            document = {
                "url": url,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            }
            opensearch_client.index(index=index_name, body=document, id=f"{url}_{i}")
            logger.debug(f"Indexed chunk {i} for URL: {url}")
        else:
            logger.warning(f"Skipping chunk {i} for URL: {url} due to missing embedding")
            has_unindexed_chunks = True
    return has_unindexed_chunks


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


def check_and_throttle(tokens_to_add, token_data):
    """
    Checks the token count and throttles if the limit is approaching.
    :param tokens_to_add: Number of tokens to add.
    :param token_data: Dictionary containing token tracking data.
    """
    # ensure token count stays within limits
    if token_data['total_tokens_used'] + tokens_to_add > TOKEN_LIMIT_PER_MINUTE:
        current_time = time.time()
        elapsed_time = current_time - token_data['last_reset_time']
        if elapsed_time < SECONDS_IN_MINUTE:
            time_to_wait = SECONDS_IN_MINUTE - elapsed_time
            print(f"Rate limit approaching, waiting for {time_to_wait:.2f} seconds.")
            time.sleep(time_to_wait)  # Sleep until the rate limit resets
        
        # reset the token counter after waiting
        token_data['total_tokens_used'] = 0
        token_data['last_reset_time'] = time.time()

    # update token count
    token_data['total_tokens_used'] += tokens_to_add

def process_urls(index_name, urls, embedding_model="text-embedding-3-small"):
    """
    Processes the list of URLs by scraping, splitting, embedding, and indexing the text content.
    :param index_name: Name of the OpenSearch index.
    :param urls: List of URLs to process.
    :param embedding_model: OpenAI embedding model to use.
    """
    urls_unindexed_chunks = []
    unindexed_chunks_file = "unindexed_chunks.json"
    
    # token tracking data
    token_data = {'total_tokens_used': 0, 'last_reset_time': time.time()}

    for url in tqdm(urls):
        logger.info(f"Processing URL: {url}")
        
        # scraping text
        text = scrape_text(url)  
        if text:
            # split the text into chunks
            chunks = split_text_with_char_splitter(text)
            # calculate the number of tokens for these chunks
            tokens_needed = calculate_tokens_for_chunks(chunks)
            # check and throttle if we exceed the token limit
            check_and_throttle(tokens_needed, token_data)
            
            try:
                embeddings = embed_text_with_openai(chunks, embedding_model)
            except APIError as e:
                # errors related to insufficient balance or any other API errors
                if "insufficient balance" in str(e).lower():
                    logger.error("Insufficient balance for OpenAI embedding. Stopping further requests. Exact error: {e}")
                    break  # Stop processing further if balance is insufficient
                else:
                    logger.error(f"Error embedding text with OpenAI API: {e}")
                    continue

            # index the embeddings
            has_unindexed_chunks = index_embeddings(index_name, chunks, embeddings, url)
            if has_unindexed_chunks:
                urls_unindexed_chunks.append(url)
    
    # write the unindexed URLs to a file if there are any
    if urls_unindexed_chunks:
        logger.warning(f"Writing {len(urls_unindexed_chunks)} urls with unindexed chunks to file: {unindexed_chunks_file}")
        try:
            with open(unindexed_chunks_file, 'w') as file:
                json.dump(urls_unindexed_chunks, file)
        except json.JSONDecodeError as e:
            logger.error(f"Error writing unindexed chunks to file: {e}")

if __name__ == "__main__":
    urls = read_urls_from_file(URLS_FILE)
    if urls:
        logger.info("Starting URL processing pipeline")
        has_config = False
        index_name = "eur-lex-diversified-knowledge-base-3"
        if has_config:
            chunk_size = 400
            chunk_overlap = 50
            text_splitter_name = "RecursiveCharacterTextSplitter"
            embedding_model = "mxbai-embed-large"
            if embedding_model == "mxbai-embed-large":
                ollama.pull("mxbai-embed-large")
                embedding_dim = 1024
            # initialize text splitter
            if text_splitter_name == "RecursiveCharacterTextSplitter":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                raise ValueError(f"Unsupported text splitter: {text_splitter_name}")
        else:
            # embedding dimension for text-embedding-3-small
            embedding_dim =  1536

        # create an OpenSearch client
        logger.info("Creating OpenSearch client.")
        opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)
        # create an index mapping
        index_mapping = {
            "mappings": {
                "properties": {
                    "url": {
                        "type": "keyword"
                    },
                    "chunk_id": {
                        "type": "integer"
                    },
                    "text": {
                        "type": "text",
                        "fielddata": False 
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim
                    }
                }
            }
        }

        # create an index if necessary
        create_index(opensearch_client, index_name, index_mapping, logger)
        # process URLs
        process_urls(index_name, urls)
        logger.info("URL processing pipeline completed")
    else:
        logger.warning("No URLs to process")
