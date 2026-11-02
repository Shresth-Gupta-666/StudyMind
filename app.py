import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="StudyMind", page_icon="📘")
st.title("📘 StudyMind")
st.caption("An adaptive RAG-based study companion — ask questions grounded in your own course material.")

PDF_PATH = "BCSE301L_SOFTWARE-ENGINEERING_TH_1.0_67_BCSE301L.pdf"


@st.cache_resource(show_spinner="Loading and indexing document...")
def load_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        match = re.search(r"Module[:\s]*(\d+)", chunk.page_content)
        chunk.metadata["module"] = match.group(1) if match else "none"

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")


vectorstore = load_vectorstore()


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
            if module_match:
                mod_tag = module_match.group(1)
            else:
                mod_search = re.search(r"Module[:\s]*(\d+)", r.page_content)
                mod_tag = mod_search.group(1) if mod_search else "none"
            label = f"Page {page_display}" + (f", Module {mod_tag}" if mod_tag != "none" else "")
            if label not in seen:
                citations.append(label)
                seen.add(label)

    return answer, citations


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

if question := st.chat_input("Ask a question about the syllabus..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, citations = ask(question)
        st.write(answer)
        if citations:
            st.caption(f"Sources: {', '.join(citations)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": citations}
    )