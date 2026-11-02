import os
import re
import json
import shutil

os.environ["HF_HUB_OFFLINE"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==================== CONFIGURATION ====================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Check your .env file."
    )

client = Groq(api_key=API_KEY)

MODEL_NAME = "openai/gpt-oss-20b"
DB_PATH = "./chroma_db"
UPLOAD_DIR = "./uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="StudyMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=80,
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Reuse the existing Chroma database.
# Do not delete or rebuild your existing database.
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)


# ==================== REQUEST MODELS ====================

class Question(BaseModel):
    question: str
    source: str | None = None


class QuizRequest(BaseModel):
    source: str | None = None
    count: int = 5


# ==================== HELPERS ====================

def groq_completion(prompt, max_tokens=700):
    """Generate a concise response and report API errors clearly."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
            max_completion_tokens=max(max_tokens, 1500),
            reasoning_effort="low",
            include_reasoning=False,
        )

        answer = response.choices[0].message.content
        if not answer:
            raise ValueError("The AI returned an empty response.")

        return answer.strip()

    except Exception as exc:
        print("Groq API error:", repr(exc))

        error_text = str(exc).lower()

        if "rate_limit" in error_text or "429" in error_text:
            message = (
                "Groq rate limit reached. Wait for the limit to reset "
                "and try again. Reduce usage if the token limit persists."
            )
        elif "401" in error_text or "authentication" in error_text:
            message = (
                "Groq authentication failed. Check GROQ_API_KEY "
                "in your .env file."
            )
        elif "model" in error_text and (
            "not found" in error_text or "decommissioned" in error_text
        ):
            message = (
                "The configured Groq model is unavailable. "
                "Check the models available to your Groq account."
            )
        else:
            message = (
                "The AI request failed. Check the backend terminal, "
                "internet connection, Groq API key, and model availability."
            )

        raise HTTPException(status_code=502, detail=message)


def get_documents(question, source=None, k=4):
    """Retrieve relevant PDF passages, optionally from one uploaded file."""

    try:
        search_kwargs = {"k": k}

        if source:
            search_kwargs["filter"] = {"source": source}

        return vectorstore.similarity_search(
            question,
            **search_kwargs,
        )

    except Exception as exc:
        print("Document search error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Document search failed. Check the Chroma database.",
        )


def format_citations(docs):
    """Return readable document names and page numbers."""

    citations = []
    seen = set()

    for doc in docs:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "Unknown document")

        if isinstance(page, int):
            page_label = f"Page {page + 1}"
        else:
            page_label = "Page unknown"

        filename = os.path.basename(source)
        label = f"{filename}: {page_label}"

        if label not in seen:
            citations.append(label)
            seen.add(label)

    return citations


def extract_json(text):
    """Extract JSON even if the model adds Markdown fences."""

    text = text.strip()
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in AI response.")

    return json.loads(text[start:end + 1])


# ==================== ROUTES ====================

@app.get("/")
def root():
    return {
        "status": "StudyMind backend running",
        "model": MODEL_NAME,
    }


@app.post("/ask")
def ask_endpoint(q: Question):
    question = q.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    docs = get_documents(
        question,
        source=q.source,
        k=4,
    )

    if not docs:
        return {
            "answer": "I couldn't find that in the document.",
            "citations": [],
        }

    # Limit context to prevent unnecessarily large API requests.
    context = "\n\n".join(
        doc.page_content[:700]
        for doc in docs
    )

    prompt = f"""Answer using ONLY the document context below.
If the answer is not supported by the context, say:
"I couldn't find that in the document."

Keep the answer concise. Preserve important formulas and
technical terms. Do not invent facts.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    answer = groq_completion(
        prompt,
        max_tokens=500,
    )

    return {
        "answer": answer,
        "citations": format_citations(docs),
    }


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    filename = os.path.basename(file.filename)
    path = os.path.abspath(os.path.join(UPLOAD_DIR, filename))

    try:
        with open(path, "wb") as output:
            shutil.copyfileobj(file.file, output)

        pages = PyPDFLoader(path).load()

        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No pages were found in the PDF.",
            )

        if not any(page.page_content.strip() for page in pages):
            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text found. This may be a scanned PDF "
                    "that requires OCR."
                ),
            )

        chunks = splitter.split_documents(pages)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract readable text from the PDF.",
            )

        for chunk in chunks:
            chunk.metadata["source"] = path

            match = re.search(
                r"Module[:\s]*(\d+)",
                chunk.page_content,
                re.IGNORECASE,
            )
            chunk.metadata["module"] = (
                match.group(1) if match else "none"
            )

        # Replace previous chunks for this exact uploaded file.
        try:
            vectorstore._collection.delete(
                where={"source": path}
            )
        except Exception as exc:
            print("Existing-chunk cleanup warning:", repr(exc))

        vectorstore.add_documents(chunks)

        return {
            "filename": filename,
            "source": path,
            "pages": len(pages),
            "chunks": len(chunks),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("PDF upload error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not process the PDF. Check the backend terminal."
            ),
        )

    finally:
        file.file.close()


@app.post("/quiz")
def generate_quiz(q: QuizRequest):
    count = max(3, min(q.count, 5))

    docs = get_documents(
        "important concepts definitions examples",
        source=q.source,
        k=4,
    )

    if not docs:
        raise HTTPException(
            status_code=404,
            detail="No document content found. Upload a PDF first.",
        )

    # Keep evidence short to reduce Groq token consumption.
    evidence = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n"
        f"{doc.page_content[:500]}"
        for doc in docs
    )

    prompt = f"""Create exactly {count} multiple-choice questions
using ONLY the evidence below.

Return a valid JSON object in this format:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": 0,
      "explanation": "Short explanation"
    }}
  ]
}}

Rules:
- Generate exactly {count} questions.
- Give exactly four string options per question.
- The answer must be an integer from 0 to 3.
- Include a short explanation based only on the evidence.
- Do not use outside knowledge.
- Return JSON only, without Markdown fences.

EVIDENCE:
{evidence}
"""

    raw = groq_completion(
        prompt,
        max_tokens=1600,
    )

    try:
        result = extract_json(raw)
        questions = result.get("questions", [])

        if not isinstance(questions, list):
            raise ValueError("The questions field is not a list.")

        valid_questions = []

        for item in questions:
            if not isinstance(item, dict):
                continue

            question_text = item.get("question")
            options = item.get("options")
            answer = item.get("answer")
            explanation = item.get("explanation", "")

            if (
                isinstance(question_text, str)
                and question_text.strip()
                and isinstance(options, list)
                and len(options) == 4
                and all(
                    isinstance(option, str) and option.strip()
                    for option in options
                )
                and type(answer) is int
                and answer in range(4)
                and isinstance(explanation, str)
            ):
                valid_questions.append({
                    "question": question_text.strip(),
                    "options": options,
                    "answer": answer,
                    "explanation": explanation,
                })

        if not valid_questions:
            raise ValueError("No valid questions were returned.")

        return {
            "questions": valid_questions[:count],
        }

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print("Quiz parsing error:", repr(exc))
        print("AI response:", raw[:1200])

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI returned an invalid quiz response. "
                "Try again. Check the backend terminal if it persists."
            ),
        )
