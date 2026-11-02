# StudyMind – AI-Powered Learning Assistant

## Project Overview
StudyMind is an AI-powered learning platform that helps students learn from their study materials. Users can upload PDF documents, ask questions based on their content, and generate interactive quizzes.

## Features
- Upload PDF study materials.
- Extract and process text from PDFs.
- Retrieve relevant information using Retrieval-Augmented Generation (RAG).
- Ask questions and receive AI-generated answers with source page references.
- Generate multiple-choice quizzes.
- Submit quizzes and view scores.

## Technology Stack
- **Frontend:** React, Vite, JavaScript
- **Backend:** Python, FastAPI
- **AI:** Groq API
- **Document Processing:** LangChain, PyPDFLoader
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)

## Project Structure
- `frontend/` – React frontend application
- `backend.py` – Backend functionality
- `app.py` – Application/backend entry point
- `test_*.py` – Supporting test scripts

## Running the Project
1. Clone this repository.
2. Install the required Python dependencies.
3. Configure your Groq API key in a `.env` file using `GROQ_API_KEY`.
4. Start the FastAPI backend.
5. Start the React frontend.
6. Open the frontend in your browser.

**Note:** Do not upload API keys, `.env` files, virtual environments, or generated database files to GitHub.

## Current Limitations
- Uploaded PDFs may need to be uploaded again after refreshing the application.
- The current version is designed as a locally running prototype.

## Future Enhancements
- Persistent document storage.
- User authentication.
- Customizable quizzes.
- Improved document management.
- Cloud deployment.
