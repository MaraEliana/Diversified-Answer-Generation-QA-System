import ollama
import logging
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import OpenSearchVectorSearch
from dotenv import load_dotenv
from opensearchpy import RequestsHttpConnection
import os
from utils import create_opensearch_client

LOG_FILE = "rag.log"
# load environment variables from the .env file
load_dotenv()
opensearch_user = os.getenv('OPENSEARCH_USER')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
openai_api_key = os.getenv('OPENAI_API_KEY')

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

def create_simple_retriever(index_name, opensearch_url, k=50):
    """
    Create a simple retriever using OpenSearch as the vector store.
    :param index_name: The name of the OpenSearch index to use for retrieval.
    :param openserach_client: The OpenSearch client to use for retrieval.
    :param k: The number of documents to retrieve.
    """
    embeddings = OpenAIEmbeddings(openai_api_key= openai_api_key, model="text-embedding-3-small")
    
    vector_store = OpenSearchVectorSearch(
        index_name=index_name, 
        embedding_function=embeddings,
        vector_field="embedding",
        opensearch_url=opensearch_url,
        http_auth=(opensearch_user, opensearch_password),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    return vector_store

def create_mmr_retriever(index_name, opensearch_client, k=50, lambda_mmr=0.5):
    """
    Create a Langchain MMR Retriever using OpenSearch as the vector store.
    :param index_name: The name of the OpenSearch index to use for retrieval.
    :param k: The number of documents to retrieve.
    """
    # Initialize OpenAI embeddings with the provided API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    # Create the OpenSearch vector store with the embeddings
    vector_store = OpenSearchVectorSearch(
        client=opensearch_client,
        index_name=index_name,
        embedding_function=embeddings.embed_query,  # Use embeddings to process queries
    )

    # Set up the MMR retriever with the OpenSearch vector store
    mmr_retriever = MMRRetriever(
        vector_store=vector_store,
        k=k,  # Number of documents to retrieve
        lambda_mmr=lambda_mmr  # Diversity parameter
    )
    
    return mmr_retriever


def generate_ollama(question, retriever):
    prompt_template = """Question: {question}
    The following documents provide relevant information: {context}
    Please answer the question, but make sure to provide a diversified response that covers different perspectives and details from the provided documents. Your answer should include multiple viewpoints and insights from the context, not just a single perspective. If necessary, highlight different interpretations, opinions, or additional context that is relevant to the question.
    Answer the question comprehensively, using the information from the documents provided.
    """
    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
    ollama.pull("llama3.3")
    llm = Ollama(model="llama3.3")

    # initialize the chain for retrieving and answering
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
    )

    # run the chain to get the response
    result = retrieval_qa({"query": question})
    return result['result'] #, result['source_documents']

def generate_openai(question, retriever, model="gpt-3.5-turbo"):
    prompt_template = """Question: {question}
    The following documents provide relevant information: {context}
    Please answer the question, but make sure to provide a diversified response that covers different perspectives and details from the provided documents. Your answer should include multiple viewpoints and insights from the context, not just a single perspective. If necessary, highlight different interpretations, opinions, or additional context that is relevant to the question.
    Answer the question comprehensively, using the information from the documents provided.
    """
    
    # Define the prompt template with placeholders for the question and context
    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
    
    # Initialize the OpenAI model (replace 'gpt-3.5-turbo' with your desired model)
    llm = OpenAI(openai_api_key=openai_api_key, model=model)
    
    # Initialize the chain for retrieving and answering
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
    )

    # Run the chain to get the response
    result = retrieval_qa({"query": question})
    
    return result['result']

def load_questions_and_answers_from_opensearch(qa_index_name, opensearch_client, size=1):
    query = {
        "query": {
            "match_all": {}
        },
        "size": size
    }

    response = opensearch_client.search(index=qa_index_name, body=query)

    questions_and_answers = []
    for hit in response['hits']['hits']:
        question = hit['_source']['question']
        answer = hit['_source']['answer']
        questions_and_answers.append((question, answer))
    
    return questions_and_answers

if __name__ == "__main__":
    # initialize OpenSearch client
    opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)
    opensearch_url = "https://opensearch-ds-2.ifi.uni-heidelberg.de:443"

    kb_index_name = "eur-lex-diversified-knowledge-base-2"
    qa_index_name = "eur-lex-diversified-qa-askep"
    k = 3
    lambda_mmr = 0.5
    nr_questions = 3
    model = "gpt-4-turbo"

    # create a retriever
    # FIRST ATTEMPT: Simple retriever
    logger.info("Creating simple retriever...")
    simple_retriever = create_simple_retriever(kb_index_name, opensearch_url, k=k)
    logger.info("Simple retriever created.")

    # SECOND ATTEMPT: MMR retriever
    # mmr_retriever = create_mmr_retriever(kb_index_name, opensearch_client, k=k, lambda_mmr=lambda_mmr)

    # load questions and ground truth answers from OpenSearch
    logger.info("Loading questions and answers from OpenSearch...")
    questions_and_answers = load_questions_and_answers_from_opensearch(qa_index_name, opensearch_client, size=nr_questions)
    logger.info("Questions and answers loaded.")
    
    for question, ground_truth_answer in questions_and_answers:
        docs = simple_retriever.similarity_search(query=question, k=k, vector_field="embedding")
        for doc in docs:
            print(doc)
        # print(f"Question: {question}")
        # print("Ground Truth Answer:")
        # for key, value in ground_truth_answer.items():
        #     print(f"{key} /n, {value}")

        # # query the retriever and generate the answer
        # # answer = generate_ollama(question, simple_retriever)
        # logger.info("Generating answer...")
        # answer = generate_openai(question, simple_retriever, model=model)
        
        # print(f"Generated Answer: {answer}")
        # print("=" * 50)