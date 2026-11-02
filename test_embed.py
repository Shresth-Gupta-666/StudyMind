from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load and chunk (same as before)
loader = PyPDFLoader("BCSE301L_SOFTWARE-ENGINEERING_TH_1.0_67_BCSE301L.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

print("Loading embedding model (first run downloads it, may take a minute)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding and storing chunks in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Done! Now let's test retrieval:")
query = "What is taught in module 3?"
results = vectorstore.similarity_search(query, k=2)

print(f"\n---- Top 2 results for: '{query}' ----")
for i, r in enumerate(results):
    print(f"\nResult {i+1}:")
    print(r.page_content)