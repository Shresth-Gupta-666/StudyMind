from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("BCSE301L_SOFTWARE-ENGINEERING_TH_1.0_67_BCSE301L.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} page(s)")
print("---- First 300 characters of page 1 ----")
print(pages[0].page_content[:300])