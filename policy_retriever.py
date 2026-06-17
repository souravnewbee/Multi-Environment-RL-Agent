"""
UMORDA — RAG Retriever for Hospital Knowledge Base
Loads policy markdown files, splits them into chunks, and retrieves the
most relevant chunk(s) for a given query using TF-IDF cosine similarity.

No external vector DB needed — the knowledge base is small enough that
TF-IDF retrieval is fast, accurate, and dependency-light.
"""

import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PolicyRetriever:
    """
    Retrieves relevant policy text chunks from the knowledge_base/ folder.

    Usage
    -----
        retriever = PolicyRetriever("knowledge_base")
        chunks = retriever.retrieve("free beds low, many patients waiting", top_k=2)
    """

    def __init__(self, kb_dir="knowledge_base"):
        self.kb_dir   = kb_dir
        self.chunks   = []   # list of {"text": ..., "source": ...}
        self._load_and_chunk()
        self._build_index()

    # ── Load all markdown files and split into paragraph-level chunks ────────
    def _load_and_chunk(self):
        if not os.path.isdir(self.kb_dir):
            raise FileNotFoundError(f"Knowledge base folder not found: {self.kb_dir}")

        for fname in sorted(os.listdir(self.kb_dir)):
            if not fname.endswith(".md"):
                continue
            path = os.path.join(self.kb_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # Split on markdown headers (##) so each chunk is one policy section
            sections = re.split(r"\n##\s+", text)
            for section in sections:
                section = section.strip()
                if not section or len(section) < 20:
                    continue
                # Re-attach header marker for readability if it was stripped
                if not section.startswith("#"):
                    section = "## " + section
                self.chunks.append({"text": section, "source": fname})

    # ── Build TF-IDF index over all chunks ───────────────────────────────────
    def _build_index(self):
        corpus = [c["text"] for c in self.chunks]
        self.vectorizer   = TfidfVectorizer(stop_words="english")
        self.doc_matrix   = self.vectorizer.fit_transform(corpus)

    # ── Retrieve top_k most relevant chunks for a query ──────────────────────
    def retrieve(self, query, top_k=2, source_filter=None):
        """
        Parameters
        ----------
        query         : str, natural language description of the situation
        top_k         : int, number of chunks to return
        source_filter : str or None, restrict search to a specific filename
                         (e.g. "bed_allocation_policy.md") for speed/precision

        Returns
        -------
        list of {"text": ..., "source": ..., "score": float}
        """
        query_vec = self.vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self.doc_matrix)[0]

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in ranked:
            if source_filter and chunk["source"] != source_filter:
                continue
            results.append({
                "text":   chunk["text"],
                "source": chunk["source"],
                "score":  float(score),
            })
            if len(results) >= top_k:
                break

        return results


# ── Task → knowledge base file mapping ────────────────────────────────────────
TASK_SOURCE_MAP = {
    "bed_allocation":   "bed_allocation_policy.md",
    "er_queue":         "er_queue_policy.md",
    "staff_allocation": "staff_allocation_policy.md",
}


def build_query(task, state, action):
    """Turn a (task, state, action) triple into a natural language query for retrieval."""
    if task == "bed_allocation":
        return (
            f"free beds {state['free_beds']} waiting patients "
            f"{state['waiting_patients']} action {action} bed allocation transfer admit reject"
        )
    elif task == "er_queue":
        return (
            f"emergency queue {state['emergency_queue']} normal queue "
            f"{state['normal_queue']} action {action} triage priority"
        )
    elif task == "staff_allocation":
        return (
            f"available doctors {state['available_doctors']} patient load "
            f"{state['patient_load']} action {action} staffing ratio"
        )
    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    # Quick smoke test
    retriever = PolicyRetriever("knowledge_base")
    print(f"Loaded {len(retriever.chunks)} chunks from knowledge base.\n")

    test_query = build_query(
        "bed_allocation",
        {"free_beds": 2, "waiting_patients": 25},
        "Transfer",
    )
    results = retriever.retrieve(test_query, top_k=2, source_filter="bed_allocation_policy.md")
    for r in results:
        print(f"[{r['source']}] score={r['score']:.3f}")
        print(r["text"][:200], "...\n")