from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from timeit_decorator import timeit

from src.constants import EMBEDDING_MODEL_NAME, AWSDOCS_INDEX

load_dotenv()

embedding = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
# chat_llm = ChatOllama(model="llama3", verbose=True)
chat_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=1
)
vector_store = PineconeVectorStore(index_name=AWSDOCS_INDEX, embedding=embedding)


@timeit(runs=1, workers=1)
def ask(question: str, chat_history: List[Dict[str, Any]] = []):
    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_docs_chain = create_stuff_documents_chain(chat_llm, qa_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(llm=chat_llm, retriever=vector_store.as_retriever(),
                                                             prompt=rephrase_prompt)
    qa_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_docs_chain)

    print(f"Asking AI the question... {question=}")
    result = qa_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    print(f"Got the answer: {result=}")

    return result


if __name__ == "__main__":
    question = "What are the the pillars for a well-architected workload?"
    result = ask(question)

    print("====================== Answer ======================")
    print(result["answer"])
    print("All done ✨🍰 ✨!")
