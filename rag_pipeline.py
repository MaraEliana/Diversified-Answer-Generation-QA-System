import ollama
import logging
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_community.vectorstores import OpenSearchVectorSearch
from dotenv import load_dotenv
from opensearchpy import RequestsHttpConnection
import os
import json
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

def create_simple_retriever(index_name, opensearch_url, k=5):
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
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k, "vector_field": "embedding"})

def create_mmr_retriever(index_name, opensearch_client, k=50, lambda_mmr=0.5):
    """
    Create a Langchain MMR Retriever using OpenSearch as the vector store.
    :param index_name: The name of the OpenSearch index to use for retrieval.
    :param k: The number of documents to retrieve.
    """
    pass


def generate_ollama(question, retriever):
    pass

def generate_openai(question, retriever, model="gpt-3.5-turbo"):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Question: {question}
        The following documents provide relevant information: {context}
        Please answer the question only by using the provided information. Make sure to provide a diversified response that covers different perspectives and details from the provided documents. Your answer should include multiple viewpoints and insights from the context, not just a single perspective. If necessary, highlight different interpretations, opinions, or additional context that is relevant to the question.
        Answer the question comprehensively, using the information from the documents provided.
        """
    )
    
    llm = ChatOpenAI(openai_api_key = openai_api_key, model=model)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
        verbose=True
    )

    response = qa_chain.invoke({"query": question})
    
    return response

def load_questions_and_answers_from_opensearch(qa_index_name, opensearch_client, size=1000):
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


def format_ground_truth(answer_dict):
    """Concatenates key-value pairs from a dictionary into a single string."""
    return " ".join(f"{k} {v}" for k, v in answer_dict.items())


if __name__ == "__main__":
    opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)
    opensearch_url = "https://opensearch-ds-2.ifi.uni-heidelberg.de:443"

    kb_index_name = "eur-lex-diversified-knowledge-base-3"
    qa_index_name = "eur-lex-diversified-qa-askep"
    k = 3
    lambda_mmr = 0.5
    nr_questions = 1000
    model = "gpt-4o-mini"
    output_dir = "qa_results_simple_retriever"

    # create a retriever
    # FIRST SCENARIO: Simple retriever
    logger.info("Creating simple retriever...")
    simple_retriever = create_simple_retriever(kb_index_name, opensearch_url, k=k)
    logger.info("Simple retriever created.")

    # load questions and ground truth answers from OpenSearch
    logger.info("Loading questions and answers from OpenSearch...")
    questions_and_answers = load_questions_and_answers_from_opensearch(qa_index_name, opensearch_client, size=nr_questions)
    logger.info("Questions and answers loaded.")
    
    results = []
    for i, (question, ground_truth_answer) in enumerate(questions_and_answers):
        logger.info(f"Processing question {i+1}...")
        # generate an answer using the simple retriever
        response = generate_openai(question, simple_retriever, model=model)
        # retrieve the source documents and the generated answer
        retrieved_docs = [doc.page_content for doc in response["source_documents"]]
        generated_answer = response["result"]
        # format the ground truth answer
        formatted_ground_truth = format_ground_truth(ground_truth_answer)
        # Save each QA pair to a JSON file
        output_file = os.path.join(output_dir, f"qa_pair_{i+1}.json")
        logger.info(f"Saving QA pair to {output_file}...")
        with open(output_file, "w") as f:
            json.dump({
                "question": question,
                "ground_truth_answer": formatted_ground_truth,
                "retrieved_documents": retrieved_docs,
                "generated_answer": generated_answer
            }, f, indent=4)

    logger.info("All questions processed.")
       