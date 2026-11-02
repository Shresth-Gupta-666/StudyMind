import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function normaliseCitation(c) {
  if (typeof c === "string") return { text: c };
  return {
    text: c.text || c.content || c.page_content || JSON.stringify(c),
    source: c.source || c.metadata?.source,
    page: c.page ?? c.metadata?.page,
  };
}

function renderInline(text = "") {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      part
    )
  );
}

function Citation({ c }) {
  const label = [
    c.source ? c.source.split(/[\\/]/).pop() : null,
    c.page !== undefined && c.page !== null
      ? `page ${Number(c.page) + 1 || c.page}`
      : null,
  ]
    .filter(Boolean)
    .join(", ");

  return (
    <li className="citation">
      <span className="mark">{c.text}</span>
      {label && <span className="where">{label}</span>}
    </li>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [thread, setThread] = useState([]);
  const [loading, setLoading] = useState(false);
  const [upload, setUpload] = useState({ state: "idle", message: "" });
  const [scope, setScope] = useState(null);
  const [quiz, setQuiz] = useState([]);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [quizLoading, setQuizLoading] = useState(false);
  const [quizError, setQuizError] = useState("");
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const fileRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [thread, loading, quiz]);

  async function ask() {
    const q = question.trim();
    if (!q || loading) return;

    setQuestion("");
    setLoading(true);
    setThread((t) => [...t, { q, pending: true }]);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, source: scope?.source ?? null }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Server returned ${res.status}`);
      }
      const citations = (data.citations || []).map(normaliseCitation);
      setThread((t) => {
        const copy = [...t];
        copy[copy.length - 1] = { q, answer: data.answer, citations };
        return copy;
      });
    } catch (err) {
      setThread((t) => {
        const copy = [...t];
        copy[copy.length - 1] = {
          q,
          error:
            err instanceof TypeError
              ? "Can't reach the backend. Start it with: python -m uvicorn backend:app --reload"
              : err.message,
        };
        return copy;
      });
    } finally {
      setLoading(false);
    }
  }

  async function uploadPdf(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUpload({ state: "error", message: "Please choose a PDF file." });
      return;
    }

    setUpload({ state: "busy", message: `Reading ${file.name}…` });
    setQuiz([]);
    setQuizError("");

    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Server returned ${res.status}`);
      }

      if (data.source) setScope({ source: data.source, name: data.filename });
      setUpload({
        state: "done",
        message: `Added ${data.filename} (${data.pages} pages).`,
      });
    } catch (err) {
      setUpload({
        state: "error",
        message:
          err instanceof TypeError
            ? "Can't reach the backend. Is uvicorn running?"
            : err.message,
      });
    }
  }

  async function generateQuiz() {
    setQuizLoading(true);
    setQuizError("");
    setQuiz([]);
    setSelectedAnswers({});
    setQuizSubmitted(false);

    try {
      const res = await fetch(`${API_URL}/quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          count: 5,
          ...(scope?.source ? { source: scope.source } : {}),
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Quiz generation failed: ${res.status}`);
      }

      const questions = Array.isArray(data)
        ? data
        : data.questions || data.quiz || [];

      if (!questions.length) {
        throw new Error("No questions returned. Upload a PDF first.");
      }

      setQuiz(questions);
    } catch (err) {
      setQuizError(
        err instanceof TypeError
          ? "Cannot reach the backend. Check that FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setQuizLoading(false);
    }
  }

  function submitQuiz() {
    setQuizSubmitted(true);
  }

  function getQuizScore() {
    return quiz.reduce(
      (score, q, i) => score + (selectedAnswers[i] === q.answer ? 1 : 0),
      0
    );
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  }

  return (
    <div className="shell">
      <header className="top">
        <h1>StudyMind</h1>
        <p>Ask your notes and textbooks. Every answer shows the passages it came from.</p>

        <div className="upload">
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            onChange={uploadPdf}
            hidden
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={upload.state === "busy"}
          >
            {upload.state === "busy" ? "Adding…" : "Add a PDF"}
          </button>
          {upload.message && (
            <span className={`upload-msg ${upload.state}`} role="status">
              {upload.message}
            </span>
          )}
        </div>

        <div className="quiz-actions">
          <button
            type="button"
            onClick={generateQuiz}
            disabled={quizLoading || upload.state === "busy"}
          >
            {quizLoading ? "Generating Quiz…" : "Take a Quiz"}
          </button>
          {quiz.length > 0 && (
            <button
              type="button"
              onClick={generateQuiz}
              disabled={quizLoading}
            >
              Generate Another
            </button>
          )}
        </div>

        {scope && (
          <div className="scope">
            Searching only <strong>{scope.name}</strong>
            <button
              type="button"
              onClick={() => setScope(null)}
              aria-label="Search all documents"
            >
              Search all documents
            </button>
          </div>
        )}
      </header>

      <main className="thread" aria-live="polite">
        {thread.length === 0 && (
          <div className="empty">
            <p>Ask a question about your uploaded material.</p>
          </div>
        )}

        {thread.map((turn, i) => (
          <section className="turn" key={i}>
            <h2 className="q">{turn.q}</h2>
            {turn.error && <p className="error">{turn.error}</p>}
            {turn.answer && <p className="answer">{renderInline(turn.answer)}</p>}
            {turn.citations?.length > 0 && (
              <div className="sources">
                <h3>From your material</h3>
                <ul>
                  {turn.citations.map((c, j) => (
                    <Citation c={c} key={j} />
                  ))}
                </ul>
              </div>
            )}
            {turn.pending && <p className="thinking">Searching your material…</p>}
          </section>
        ))}

        <div ref={endRef} />

        {quizError && <p className="error">{quizError}</p>}

        {quizLoading && (
          <section className="quiz">
            <h2>Practice Quiz</h2>
            <p className="thinking">Generating questions from your document…</p>
          </section>
        )}

        {quiz.length > 0 && (
          <section className="quiz">
            <h2>Practice Quiz</h2>
            <p>Choose one answer for each question, then submit to see your score.</p>

            {quiz.map((q, i) => (
              <div key={i} className="quiz-question">
                <h3>{i + 1}. {q.question}</h3>
                {(q.options || []).map((option, j) => (
                  <label key={j} className="quiz-option">
                    <input
                      type="radio"
                      name={`question-${i}`}
                      disabled={quizSubmitted}
                      checked={selectedAnswers[i] === j}
                      onChange={() =>
                        setSelectedAnswers((prev) => ({ ...prev, [i]: j }))
                      }
                    />
                    {option}
                  </label>
                ))}

                {quizSubmitted && (
                  <div className="quiz-feedback">
                    <p>
                      <strong>
                        {selectedAnswers[i] === q.answer ? "Correct" : "Incorrect"}.
                      </strong>{" "}
                      Correct answer: {q.options?.[q.answer]}
                    </p>
                    {q.explanation && <p>{q.explanation}</p>}
                  </div>
                )}
              </div>
            ))}

            {!quizSubmitted ? (
              <button type="button" onClick={submitQuiz}>Submit Quiz</button>
            ) : (
              <div className="quiz-result">
                <h3>Score: {getQuizScore()} / {quiz.length}</h3>
                <button type="button" onClick={generateQuiz} disabled={quizLoading}>
                  Try Another Quiz
                </button>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="composer">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question"
          rows={2}
          aria-label="Your question"
        />
        <button onClick={ask} disabled={loading || !question.trim()}>
          {loading ? "Searching" : "Ask"}
        </button>
      </footer>
    </div>
  );
}
