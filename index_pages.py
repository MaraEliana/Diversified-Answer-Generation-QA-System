import os
import logging
import requests
import json
import ollama
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch
from utils import create_opensearch_client, create_index

# config files
CONFIG_FILE = "configurations.json"
URLS_FILE = "nonpdf_urls.json"
LOG_FILE = "index_pages.log"

# load environment variables from the .env file
load_dotenv()
opensearch_user = os.getenv('OPENSEARCH_USER')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')

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

# load configuration
def load_configuration(config_name):
    logger.info(f"Loading configuration: {config_name}")
    try:
        with open(CONFIG_FILE, 'r') as file:
            configs = json.load(file)
            if config_name in configs:
                return configs[config_name]
            else:
                raise ValueError(f"Configuration '{config_name}' not found in {CONFIG_FILE}")
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding configuration file: {e}")
        return None

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

# split text using LangChain
def split_text_with_langchain(text, text_splitter):
    logger.info(f"Splitting text into chunks.")
    return text_splitter.split_text(text)

# embed text chunks using Ollama
def embed_text_with_ollama(embedding_model, chunks):
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

# index embeddings in OpenSearch
def index_embeddings(index_name, chunks, embeddings, url):
    """
    Embeds and indexes the text chunks in OpenSearch.
    :param index_name: Index name.
    :param chunks: List of text chunks.
    :param embeddings: List of embeddings corresponding to the text chunks.
    :param url: URL of the page.
    :return: List of unindexed chunks with corresponding chunk_id and url.
    """
    unindexed_chunks = []
    logger.info(f"Indexing embeddings for URL: {url}.")
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if embedding:
            document = {
                "url": url,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            }
            opensearch_client.index(index=index_name, body=document)
            logger.debug(f"Indexed chunk {i} for URL: {url}")
        else:
            logger.warning(f"Skipping chunk {i} for URL: {url} due to missing embedding")
            unindexed_chunks.append({"url": url, "chunk_id": i, "text": chunk})
    return unindexed_chunks


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

def process_urls(index_name, urls, text_splitter, embedding_model):
    unindexed_chunks_list = []
    unindexed_chunks_file = "unindexed_chunks.json"
    for url in tqdm(urls):
        logger.info(f"Processing URL: {url}")
        text = scrape_text(url)
        if text:
            chunks = split_text_with_langchain(text, text_splitter)
            embeddings = embed_text_with_ollama(embedding_model, chunks)
            unindexed_chunks = index_embeddings(index_name, chunks, embeddings, url)
            unindexed_chunks_list.extend(unindexed_chunks)
    
    if unindexed_chunks_list:
        logger.warning(f"Writing {len(unindexed_chunks_list)} unindexed chunks to file: {unindexed_chunks_file}")
        try:
            with open(unindexed_chunks_file, 'w') as file:
                json.dump(unindexed_chunks_list, file)
        except json.JSONDecodeError as e:
            logger.error(f"Error writing unindexed chunks to file: {e}")


if __name__ == "__main__":
    urls = read_urls_from_file(URLS_FILE)
    if urls:
        logger.info("Starting URL processing pipeline")

        # load specific configuration
        # CONFIG MODEL: ChunkSize-ChunkOverlap-TextSplitter-EmbeddingModel
        config_name = "CS400-CO50-Recursive-MXBAI"
        index_name = "eur-lex-diversified-kb-" + config_name.lower()
        config = load_configuration(config_name)
        
        # extract configuration values
        chunk_size = config.get("ChunkSize")
        chunk_overlap = config.get("ChunkOverlap")
        text_splitter_name = config.get("TextSplitter")
        embedding_model = config.get("EmbeddingModel")
        if embedding_model == "mxbai-embed-large":
            ollama.pull("mxbai-embed-large")
            embedding_dim = 1024

        # create an OpenSearch client
        logger.info("Creating OpenSearch client.")
        opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)

        # initialize text splitter
        if text_splitter_name == "RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported text splitter: {text_splitter_name}")

        logger.info(f"Configuration loaded: {config}")
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
                        "type": "text"
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
        process_urls(index_name, urls, text_splitter, embedding_model)
        logger.info("URL processing pipeline completed")
    else:
        logger.warning("No URLs to process")
