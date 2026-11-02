from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("BCSE301L_SOFTWARE-ENGINEERING_TH_1.0_67_BCSE301L.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)

print(f"Split into {len(chunks)} chunks")
print("---- Chunk 1 ----")
print(chunks[0].page_content)
print("---- Chunk 2 ----")
print(chunks[1].page_content)