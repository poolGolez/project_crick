from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from timeit_decorator import timeit

from src.constants import EMBEDDING_MODEL_NAME, AWSDOCS_INDEX

load_dotenv()

embedding = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
chat_llm = ChatOllama(model="llama3", verbose=True)
vector_store = PineconeVectorStore(index_name=AWSDOCS_INDEX, embedding=embedding)


@timeit(runs=1, workers=1)
def ask(question: str):
    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # qa_prompt = ChatPromptTemplate.from_messages([
    #     ("system","You are an AWS expert. Answer questions based on the context and documentations below:\n\n<context>\n{context}\n</context>"),
    #     ("placeholder", "{chat_history}"),
    #     ("human", "{input}"),
    # ])
    stuff_docs_chain = create_stuff_documents_chain(chat_llm, qa_prompt)

    qa_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=stuff_docs_chain)
    print("Asking AI the question...")
    result = qa_chain.invoke({"input": question})

    return result


if __name__ == "__main__":
    question = "What are the the pillars for a well-architected workload?"
    result = ask(question)

    print("====================== Answer ======================")
    print(result["answer"])
    print("All done ✨🍰 ✨!")
