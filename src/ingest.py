from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = "../resources/docs/wellarchitected-framework.pdf"


def run():
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {FILE_PATH}.")

    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=150)
    chunks = recursive_splitter.split_documents(docs)
    print(f"\t> Generated {len(chunks)} chunks.")

    print("Done.")


if __name__ == "__main__":
    run()
    print("All done ✨🍰 ✨!")
