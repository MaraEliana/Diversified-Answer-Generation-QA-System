import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_community.vectorstores import OpenSearchVectorSearch
from dotenv import load_dotenv
from opensearchpy import RequestsHttpConnection
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Any
import os
import json
import numpy as np
from utils import create_opensearch_client

LOG_FILE = "rag.log"
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

def max_marginal_relevance_search(
        vector_store: VectorStore,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """
        This function is originally from the langchain. We slightly modified it at the end to ignore the metadata field.
        Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        vector_field = kwargs.get("vector_field", "vector_field")
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        # get embedding of the user query
        embedding = vector_store.embedding_function.embed_query(query)

        # do ANN/KNN search to get top fetch_k results where fetch_k >= k
        results = vector_store._raw_similarity_search_with_score_by_vector(
            embedding, fetch_k, **kwargs
        )

        embeddings = [result["_source"][vector_field] for result in results]

        # rerank top k results using MMR, (mmr_selected is a list of indices)
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        print(len(mmr_selected))

        return [
            Document(
                page_content=results[i]["_source"][text_field],
                id=results[i]["_id"],
            )
            for i in mmr_selected
        ]

def generate_openai(question, retriever, model="gpt-3.5-turbo"):
    """
    Generate an answer to a question using OpenAI's API.
    :param question: The question to answer.
    :param retriever: The retriever to use for retrieving documents.
    :param model: The OpenAI model to use for answering the question.
    """
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
    """
    Load questions and answers from an OpenSearch index.
    :param qa_index_name: The name of the OpenSearch index to load questions and answers from.
    :param opensearch_client: The OpenSearch client to use for querying the index.
    :param size: The number of questions and answers to load."""
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

def generate_with_mmr_retrieval(kb_index_name, opensearch_url, question, k=3, lambda_mmr=0.5, model="gpt-4o-mini"):
    """
    Generate an answer to a question using OpenAI's API and MMR retrieval.
    :param kb_index_name: The name of the OpenSearch index to use for retrieval.
    :param opensearch_url: The URL of the OpenSearch instance.
    :param question: The question to answer.
    :param k: The number of documents to retrieve.
    :param lambda_mmr: The lambda parameter for MMR.
    :param model: The OpenAI model to use for answering the question.
    :return: The generated answer and the retrieved documents.
    """
    embeddings = OpenAIEmbeddings(openai_api_key= openai_api_key, model="text-embedding-3-small")
    vector_store = OpenSearchVectorSearch(
        index_name=kb_index_name, 
        embedding_function=embeddings,
        vector_field="embedding",
        opensearch_url=opensearch_url,
        http_auth=(opensearch_user, opensearch_password),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    # retrieve the source documents using MMR
    retrieved_docs = max_marginal_relevance_search(vector_store, question, k=k, fetch_k=20, lambda_mult=lambda_mmr, vector_field="embedding")
    retrieved_docs = [doc.page_content for doc in retrieved_docs]
    context = "\n\n".join(retrieved_docs)
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
    chain = LLMChain(llm=llm, prompt=prompt_template)
    generated_answer = chain.run(context=context, question=question)
    return generated_answer, retrieved_docs

if __name__ == "__main__":
    opensearch_client = create_opensearch_client(username=opensearch_user, password=opensearch_password)
    opensearch_url = "https://opensearch-ds-2.ifi.uni-heidelberg.de:443"

    kb_index_name = "eur-lex-diversified-knowledge-base-3"
    qa_index_name = "eur-lex-diversified-qa-askep"
    k = 3
    lambda_mmr = 0.5
    nr_questions = 1000
    model = "gpt-4o-mini"
    output_dir = "qa_results_mmr_retriever"
    method = "mmr"

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
        if method == "simple":
            # generate an answer using the simple retriever
            response = generate_openai(question, simple_retriever, model=model)
            # retrieve the source documents and the generated answer
            retrieved_docs = [doc.page_content for doc in response["source_documents"]]
            generated_answer = response["result"]
        elif method == "mmr":
            generated_answer, retrieved_docs = generate_with_mmr_retrieval(kb_index_name, opensearch_url, question, k=k, lambda_mmr=lambda_mmr, model=model)
        # format the ground truth answer
        formatted_ground_truth = format_ground_truth(ground_truth_answer)
        # save each QA pair to a JSON file
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
       