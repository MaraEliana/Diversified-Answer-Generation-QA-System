import os
import json
import logging
import requests
from dotenv import load_dotenv
import time
from tqdm import tqdm
from markitdown import MarkItDown
from openai import OpenAI, APIError
from opensearchpy import OpenSearch
from utils import create_opensearch_client, create_index
from index_non_pdfs import split_text_with_char_splitter, calculate_tokens_for_chunks, check_and_throttle

# config files
PDF_URLS_FILE = "pdf_urls.json"
LOG_FILE = "pdfs.log"

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# load environment variables for OpenSearch
load_dotenv()
opensearch_user = os.getenv('OPENSEARCH_USER')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def download_pdf(url, save_path):
    """
    Downloads a PDF from a URL and saves it locally.
    :param url: URL of the PDF to download
    :param save_path: Local path to save the downloaded PDF
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        logger.info(f"PDF downloaded successfully: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        return False

def convert_pdf_to_markdown(pdf_path):
    """
    Converts a PDF to markdown text using MarkItDown.
    :param pdf_path: Local path to the PDF file
    :return: Markdown text content of the PDF
    """
    try:
        md = MarkItDown()
        result = md.convert(pdf_path)
        return result.text_content
    except Exception as e:
        logger.error(f"Error converting PDF to markdown: {e}")
        return None

# index embeddings in OpenSearch
def index_embeddings(index_name, chunks, embeddings, url, opensearch_client):
    """
    Embeds and indexes the text chunks in OpenSearch.
    :param index_name: Index name.
    :param chunks: List of text chunks.
    :param embeddings: List of embeddings corresponding to the text chunks.
    :param url: URL of the page.
    :param opensearch_client: OpenSearch client.
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

def embed_text_with_openai(chunks, model="text-embedding-3-small"):
    logger.info(f"Embedding {len(chunks)} chunks with OpenAI's embedding model: {model}")
    embeddings = []
    for chunk in chunks:
        embeddings.append(client.embeddings.create(input = [chunk], model=model).data[0].embedding)
    return embeddings

def process_pdfs(index_name, pdf_urls, temp_folder="temp_pdfs"):
    """
    Processes a list of PDF URLs: downloads, converts, chunks, embeds, and indexes them.
    """
    os.makedirs(temp_folder, exist_ok=True)
    urls_unindexed_chunks = []
    unindexed_chunks_file = "unindexed_chunks.json"
    
    # Token tracking data
    token_data = {'total_tokens_used': 0, 'last_reset_time': time.time()}

    for url in tqdm(pdf_urls):
        logger.info(f"Processing PDF: {url}")
        pdf_path = os.path.join(temp_folder, "temp_pdf.pdf")
        reponse = download_pdf(url, pdf_path)
        if reponse:
            markdown_content = convert_pdf_to_markdown(pdf_path)
            if markdown_content:
                # Use RecursiveCharacterTextSplitter to split the text into chunks
                chunks = split_text_with_char_splitter(markdown_content)
            
                # Calculate the number of tokens for these chunks
                tokens_needed = calculate_tokens_for_chunks(chunks)
            
                # Check and throttle if we exceed the token limit
                check_and_throttle(tokens_needed, token_data)
            
                try:
                    # Embed the text chunks using the OpenAI model
                    embeddings = embed_text_with_openai(chunks)
                except APIError as e:
                    # Catch errors related to insufficient balance or any other API errors
                    if "insufficient balance" in str(e).lower():
                        logger.error("Insufficient balance for OpenAI embedding. Stopping further requests. Exact error: {e}")
                        break  # Stop processing further if balance is insufficient
                    else:
                        logger.error(f"Error embedding text with OpenAI API: {e}")
                        continue  # Continue with the next URL
                
                # Index the embeddings
                has_unindexed_chunks = index_embeddings(index_name, chunks, embeddings, url, opensearch_client)
                if has_unindexed_chunks:
                    urls_unindexed_chunks.append(url)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info(f"Temporary file deleted: {pdf_path}")
        else:
            logger.error(f"Error processing PDF: {url}")
            os.remove(pdf_path)
    
    # Write the unindexed URLs to a file if there are any
    if urls_unindexed_chunks:
        logger.warning(f"Writing {len(urls_unindexed_chunks)} urls with unindexed chunks to file: {unindexed_chunks_file}")
        try:
            with open(unindexed_chunks_file, 'w') as file:
                json.dump(urls_unindexed_chunks, file)
        except json.JSONDecodeError as e:
            logger.error(f"Error writing unindexed chunks to file: {e}")
    

if __name__ == "__main__":
    # load PDF URLs
    with open(PDF_URLS_FILE, 'r') as file:
        pdf_urls = json.load(file)

    if pdf_urls:
        logger.info("Starting URL processing pipeline")
        # change this if everything works fine
        index_name = "eur-lex-diversified-knowledge-base-only-pdfs"
        # embedding dimension for text-embedding-3-small
        embedding_dim = 1536

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
        process_pdfs(index_name, pdf_urls)
        logger.info("URL processing pipeline completed")
    else:
        logger.warning("No URLs to process")
