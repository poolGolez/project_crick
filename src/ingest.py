from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.constants import AWSDOCS_INDEX, EMBEDDING_MODEL_NAME, AWS_DOCS_FILE_PATH

load_dotenv()


def run(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {file_path}.")

    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=150)
    chunks = recursive_splitter.split_documents(docs)
    print(f"\t> Generated {len(chunks)} chunks.")

    print("Starting embeddings")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=AWSDOCS_INDEX)
    print(f"\t> Added {len(chunks)} to Pinecone")


if __name__ == "__main__":
    run(AWS_DOCS_FILE_PATH)
    print("All done ✨🍰 ✨!")
