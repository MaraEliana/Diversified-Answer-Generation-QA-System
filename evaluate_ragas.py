import json
import os
from dotenv import load_dotenv
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, LLMContextPrecisionWithReference

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

def load_data_from_json(file_path):
    """Loads data from a single JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_data_from_multiple_files(directory_path, file_extension="json"):
    """Loads data from multiple files (either JSON or JSONL)."""
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(file_extension):
            file_path = os.path.join(directory_path, filename)
            if file_extension == "json":
                data.append(load_data_from_json(file_path))
    return data


if __name__ == "__main__":
    directory_path = "qa_results_simple_retriever"
    file_extension = "json"
    evaluation_file = "evaluation_results_simple_retriever.json"
    run_config = RunConfig(timeout=300, log_tenacity=True, max_workers=8)

    # Load the data
    print("Loading data...")
    data = load_data_from_multiple_files(directory_path, file_extension=file_extension)
    dataset = []
    for item in data:
        dataset.append(
            {
                "user_input": item["question"],
                "retrieved_contexts": item["retrieved_documents"],
                "response":item["generated_answer"],
                "reference":item["ground_truth_answer"]
            }
        )
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings(openai_api_key= openai_api_key, model="text-embedding-3-small")
    evaluator_llm = LangchainLLMWrapper(llm)
    # , FactualCorrectness(), ResponseRelevancy()
    result = evaluate(dataset=evaluation_dataset, embeddings=embeddings, metrics=[LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness()], llm=evaluator_llm, run_config=run_config)
    print(result)
    print("✅ Evaluation results saved.")