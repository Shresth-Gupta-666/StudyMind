import re
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

loader = PyPDFLoader("BCSE301L_SOFTWARE-ENGINEERING_TH_1.0_67_BCSE301L.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(pages)

# NEW: tag each chunk with its module number, if it has one
for chunk in chunks:
    match = re.search(r"Module[:\s]*(\d+)", chunk.page_content)
    chunk.metadata["module"] = match.group(1) if match else "none"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

def ask(question):
    module_match = re.search(r"module\s*(\d+)", question, re.IGNORECASE)
    candidates = vectorstore.similarity_search(question, k=15)

    if module_match:
        target = module_match.group(1)
        exact = [r for r in candidates if re.search(rf"Module[:\s]*{target}\b", r.page_content)]
        results = exact[:2] if exact else candidates[:4]
    else:
        results = candidates[:4]
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    citations = []
    if "couldn't find" not in answer.lower():
        seen = set()
        for r in results:
            page_num = r.metadata.get("page", "?")
            page_display = int(page_num) + 1 if isinstance(page_num, int) else page_num
            module_search = re.search(r"Module[:\s]*(\d+)", r.page_content)
            module_tag = module_search.group(1) if module_search else "none"
            label = f"Page {page_display}" + (
                f", Module {module_tag}" if module_tag != "none" else ""
            )
            if label not in seen:
                citations.append(label)
                seen.add(label)

    return answer, citations


def print_answer(question):
    answer, citations = ask(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    if citations:
        print(f"Sources: {', '.join(citations)}")
    print()


print_answer("What is taught in module 3?")
print("---")
print_answer("How many hours does module 5 take?")
print("---")
print_answer("What is the capital of France?")