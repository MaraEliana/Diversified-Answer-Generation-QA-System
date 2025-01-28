import ollama
from langchain.llms import Ollama
from langchain.vectorstores import OpenSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import MMRRetriever
from opensearchpy import OpenSearch
from dotenv import load_dotenv
import numpy as np
import os
from utils import create_opensearch_client

# load environment variables from the .env file
load_dotenv()
opensearch_user = os.getenv('OPENSEARCH_USER')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')

def create_mmr_retriever(index_name, k=50):
    """
    Create a Langchain MMR Retriever using OpenSearch as the vector store.
    :param index_name: The name of the OpenSearch index to use for retrieval.
    :param k: The number of documents to retrieve.
    """
    # create the vector store from OpenSearch
    vector_store = OpenSearch(opensearch_client=opensearch_client, index_name=index_name)

    # set up MMR Retriever with max documents to retrieve and lambda parameter for diversity
    mmr_retriever = MMRRetriever(
        vector_store=vector_store,
        k=k, 
        lambda_mmr=0.5
    )
    return mmr_retriever


def query_with_mmr_retriever(question, retriever):
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

def load_questions_and_answers_from_opensearch(qa_index_name, size=1):
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
    opensearch_client = create_opensearch_client(opensearch_user, opensearch_password)

    index_name = "eur-lex-diversified-kb-CS400-CO50-Recursive-MXBAI"
    qa_index_name = "eur-lex-diversified-qa-askep"

    # create MMR retriever
    mmr_retriever = create_mmr_retriever(index_name, k=50)

    # load questions and ground truth answers from OpenSearch
    questions_and_answers = load_questions_and_answers_from_opensearch(qa_index_name, size=3)
    
    for question, ground_truth_answer in questions_and_answers:
        print(f"Question: {question}")
        print("Ground Truth Answer:")
        for key, value in ground_truth_answer.items():
            print(f"{key} /n, {value}")

        # query the retriever and generate the answer
        answer = query_with_mmr_retriever(question, mmr_retriever)
        
        print("Generated Answer:")
        print(answer)
        print("=" * 50)