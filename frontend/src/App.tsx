import React, { useMemo, useState } from "react";
import { API_BASE_URL } from "./config";

type RAGAnswerChunk = {
  text: string;
  source_path: string;
  page_number?: number | null;
  doc_type?: string | null;
  score?: number | null;
};

type RAGAnswerResponse = {
  query: string;
  answer: string;
  references: RAGAnswerChunk[];
};

type VisionDetection = {
  label: string;
  conf: number;
  box_xyxy: number[];
};

type VisionHazardResponse = {
  hazard_summary: string;
  detections: VisionDetection[];
  rag_query: string;
  answer: string;
  references: RAGAnswerChunk[];
};

function extractBullets(text: string): string[] {
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  const bullets = lines
    .filter((l) => /^[-•]\s+/.test(l))
    .map((l) => l.replace(/^[-•]\s+/, "").trim())
    .filter(Boolean);

  return bullets.length >= 2 ? bullets : [];
}

function filenameFromPath(p: string): string {
  // Works for Windows "\" paths and unix "/" paths
  const parts = p.split(/[/\\]/);
  return parts[parts.length - 1] || p;
}

const App: React.FC = () => {
  // ---- Text RAG state ----
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(6);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RAGAnswerResponse | null>(null);

  const bullets = useMemo(() => {
    if (!result?.answer) return [];
    return extractBullets(result.answer);
  }, [result?.answer]);

  // ---- Vision YOLO state ----
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [visionTopK, setVisionTopK] = useState(6);
  const [visionConf, setVisionConf] = useState(0.45);
  const [visionLoading, setVisionLoading] = useState(false);
  const [visionError, setVisionError] = useState<string | null>(null);
  const [visionResult, setVisionResult] = useState<VisionHazardResponse | null>(null);

  const visionBullets = useMemo(() => {
    if (!visionResult?.answer) return [];
    return extractBullets(visionResult.answer);
  }, [visionResult?.answer]);

  // ---- Handlers ----
  const handleSubmitText = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const resp = await fetch(`${API_BASE_URL}/rag/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`API error ${resp.status}: ${text}`);
      }

      const data: RAGAnswerResponse = await resp.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message ?? "Unknown error contacting RAG backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitVision = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!imageFile) return;

    setVisionLoading(true);
    setVisionError(null);
    setVisionResult(null);

    try {
      const form = new FormData();
      form.append("file", imageFile);

      const url = new URL(`${API_BASE_URL}/vision/hazard`);
      url.searchParams.set("top_k", String(visionTopK));
      url.searchParams.set("conf", String(visionConf));

      const resp = await fetch(url.toString(), {
        method: "POST",
        body: form,
        // IMPORTANT: do not set Content-Type manually for FormData
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`API error ${resp.status}: ${text}`);
      }

      const data: VisionHazardResponse = await resp.json();
      setVisionResult(data);
    } catch (err: any) {
      setVisionError(err.message ?? "Unknown error contacting vision backend.");
    } finally {
      setVisionLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: "2rem" }}>
      <h1>Mining Safety RAG Assistant</h1>
      <p>
        Prototype mining WHS assistant. Not a substitute for legal advice, statutory obligations,
        or site supervisor instructions.
      </p>

      {/* ---------------- TEXT RAG ---------------- */}
      <div style={{ marginTop: "1.5rem" }}>
        <h2 style={{ marginBottom: "0.5rem" }}>Ask (Text)</h2>
        <form onSubmit={handleSubmitText} style={{ marginBottom: "1rem" }}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={4}
            style={{ width: "100%", padding: "0.75rem" }}
            placeholder="e.g. What are the key controls for working around mobile plant in open cut coal mines?"
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginTop: "0.5rem",
              gap: "0.75rem",
            }}
          >
            <label>
              Top K:
              <input
                type="number"
                min={3}
                max={10}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{ width: "4rem", marginLeft: "0.25rem" }}
              />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Answering…" : "Ask"}
            </button>
          </div>
        </form>

        {loading && <div style={{ marginTop: "0.5rem" }}>Answering your question…</div>}

        {error && (
          <div style={{ marginTop: "0.5rem", color: "#b00020", fontWeight: 500 }}>
            Error: {error}
          </div>
        )}

        {!loading && !error && result && (
          <div
            style={{
              marginTop: "1rem",
              padding: "1rem",
              borderRadius: "8px",
              backgroundColor: "#e8f5e9",
              color: "#111",
              border: "1px solid #c8e6c9",
            }}
          >
            <h3 style={{ marginTop: 0, color: "#111" }}>Answer</h3>
            <p style={{ color: "#111" }}>
              <strong>Question:</strong> {result.query}
            </p>

            {bullets.length > 0 ? (
              <ul style={{ paddingLeft: "1.2rem", lineHeight: 1.55, marginTop: "0.5rem" }}>
                {bullets.map((b, i) => (
                  <li key={i} style={{ marginBottom: "0.4rem" }}>
                    {b}
                  </li>
                ))}
              </ul>
            ) : (
              <div style={{ whiteSpace: "pre-wrap", marginBottom: "1rem", lineHeight: 1.55 }}>
                {result.answer}
              </div>
            )}

            <h4 style={{ marginBottom: "0.5rem", color: "#111" }}>Sources</h4>
            <ul style={{ paddingLeft: "1.1rem" }}>
              {result.references.map((ref, idx) => (
                <li key={idx} style={{ marginBottom: "0.75rem" }}>
                  <div style={{ fontSize: "0.9rem" }}>
                    <strong>[{idx + 1}]</strong> {ref.doc_type ?? "doc"} |{" "}
                    {filenameFromPath(ref.source_path)} | page {ref.page_number ?? "?"}
                  </div>
                  <div style={{ fontSize: "0.85rem", whiteSpace: "pre-wrap" }}>
                    {ref.text.slice(0, 300)}
                    {ref.text.length > 300 ? " ..." : ""}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* ---------------- VISION YOLO ---------------- */}
      <div style={{ marginTop: "2.5rem" }}>
        <h2 style={{ marginBottom: "0.5rem" }}>Assess (Image)</h2>
        <p style={{ marginTop: 0, color: "#555" }}>
          Upload a site photo. The system detects people/vehicles and returns grounded WHS controls.
        </p>

        <form onSubmit={handleSubmitVision} style={{ marginBottom: "1rem" }}>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
          />

          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginTop: "0.75rem",
              gap: "0.75rem",
              flexWrap: "wrap",
            }}
          >
            <label>
              Top K:
              <input
                type="number"
                min={3}
                max={10}
                value={visionTopK}
                onChange={(e) => setVisionTopK(Number(e.target.value))}
                style={{ width: "4rem", marginLeft: "0.25rem" }}
              />
            </label>

            <label>
              Conf:
              <input
                type="number"
                step="0.05"
                min={0.1}
                max={0.9}
                value={visionConf}
                onChange={(e) => setVisionConf(Number(e.target.value))}
                style={{ width: "5rem", marginLeft: "0.25rem" }}
              />
            </label>

            <button type="submit" disabled={visionLoading || !imageFile}>
              {visionLoading ? "Analyzing…" : "Analyze image"}
            </button>
          </div>
        </form>

        {visionLoading && <div style={{ marginTop: "0.5rem" }}>Analyzing image…</div>}

        {visionError && (
          <div style={{ marginTop: "0.5rem", color: "#b00020", fontWeight: 500 }}>
            Error: {visionError}
          </div>
        )}

        {!visionLoading && !visionError && visionResult && (
          <div
            style={{
              marginTop: "1rem",
              padding: "1rem",
              borderRadius: "8px",
              backgroundColor: "#e8f5e9",
              color: "#111",
              border: "1px solid #c8e6c9",
            }}
          >
            <h3 style={{ marginTop: 0, color: "#111" }}>Vision result</h3>

            <p style={{ color: "#111" }}>
              <strong>Hazards:</strong> {visionResult.hazard_summary}
            </p>

            <details style={{ marginBottom: "0.75rem" }}>
              <summary style={{ cursor: "pointer" }}>Detections ({visionResult.detections.length})</summary>
              <pre style={{ whiteSpace: "pre-wrap", marginTop: "0.5rem" }}>
                {JSON.stringify(visionResult.detections, null, 2)}
              </pre>
            </details>

            <h4 style={{ marginBottom: "0.5rem", color: "#111" }}>Controls checklist</h4>

            {visionBullets.length > 0 ? (
              <ul style={{ paddingLeft: "1.2rem", lineHeight: 1.55, marginTop: "0.5rem" }}>
                {visionBullets.map((b, i) => (
                  <li key={i} style={{ marginBottom: "0.4rem" }}>
                    {b}
                  </li>
                ))}
              </ul>
            ) : (
              <div style={{ whiteSpace: "pre-wrap", marginBottom: "1rem", lineHeight: 1.55 }}>
                {visionResult.answer}
              </div>
            )}

            <h4 style={{ marginBottom: "0.5rem", color: "#111" }}>Sources</h4>
            <ul style={{ paddingLeft: "1.1rem" }}>
              {visionResult.references.map((ref, idx) => (
                <li key={idx} style={{ marginBottom: "0.75rem" }}>
                  <div style={{ fontSize: "0.9rem" }}>
                    <strong>[{idx + 1}]</strong> {ref.doc_type ?? "doc"} |{" "}
                    {filenameFromPath(ref.source_path)} | page {ref.page_number ?? "?"}
                  </div>
                  <div style={{ fontSize: "0.85rem", whiteSpace: "pre-wrap" }}>
                    {ref.text.slice(0, 300)}
                    {ref.text.length > 300 ? " ..." : ""}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
