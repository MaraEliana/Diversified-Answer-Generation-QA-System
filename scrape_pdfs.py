import os
import json
import logging
import requests
from dotenv import load_dotenv
import ollama
from tqdm import tqdm
from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch
from utils import create_opensearch_client, create_index
from index_pages import split_text_with_langchain, embed_text_with_ollama, index_embeddings, load_configuration

# config files
CONFIG_FILE = "configurations.json"
PDF_URLS_FILE = "pdf_urls.json"
LOG_FILE = "index_pdfs.log"

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
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")

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


def process_pdfs(index_name, pdf_urls, text_splitter, embedding_model, temp_folder="temp_pdfs"):
    """
    Processes a list of PDF URLs: downloads, converts, chunks, embeds, and indexes them.
    """
    os.makedirs(temp_folder, exist_ok=True)
    unindexed_chunks_list = []

    for url in tqdm(pdf_urls):
        logger.info(f"Processing PDF: {url}")
        pdf_path = os.path.join(temp_folder, "temp_pdf.pdf")
        download_pdf(url, pdf_path)
        try:
            markdown_content = convert_pdf_to_markdown(pdf_path)
            if markdown_content:
                chunks = split_text_with_langchain(markdown_content, text_splitter)
                embeddings = embed_text_with_ollama(embedding_model, chunks)
                unindexed_chunks = index_embeddings(index_name, chunks, embeddings, url)
                unindexed_chunks_list.extend(unindexed_chunks)
            os.remove(pdf_path)
            logger.info(f"Temporary file deleted: {pdf_path}")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            os.remove(pdf_path)
    return unindexed_chunks_list

if __name__ == "__main__":
    # load PDF URLs
    with open(PDF_URLS_FILE, 'r') as file:
        pdf_urls = json.load(file)

    if pdf_urls:
        logger.info("Starting PDF processing pipeline")

        # load configuration
        config_name = "CS400-CO50-Recursive-MXBAI"
        index_name = "eur-lex-diversified-kb-" + config_name.lower()
        config = load_configuration(config_name)

        # Extract configuration values
        chunk_size = config["ChunkSize"]
        chunk_overlap = config["ChunkOverlap"]
        text_splitter_name = config.get("TextSplitter")
        embedding_model = config["EmbeddingModel"]
        if embedding_model == "mxbai-embed-large":
            ollama.pull("mxbai-embed-large")
            embedding_dim = 1024

        # initialize OpenSearch client
        opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)

        # initialize text splitter
        if text_splitter_name == "RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported text splitter: {text_splitter_name}")

        # index mapping
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
        create_index(opensearch_client, index_name, index_mapping, logger)

        # process the PDFs
        unindexed_chunks = process_pdfs(index_name, pdf_urls, text_splitter, embedding_model)
        if unindexed_chunks:
            with open("unindexed_chunks_pdfs.json", 'w') as file:
                json.dump(unindexed_chunks, file)
                logger.warning(f"Unindexed chunks saved to unindexed_chunks.json")
        
        logger.info("PDF processing pipeline completed")
    else:
        logger.warning("No PDF URLs found in the provided file")