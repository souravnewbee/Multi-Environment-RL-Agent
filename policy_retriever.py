"""
UMORDA — RAG Retriever for Knowledge Base (Hospital + Traffic + Energy)
Loads policy markdown files, splits them into chunks, and retrieves the
most relevant chunk(s) for a given query using TF-IDF cosine similarity.

No external vector DB needed — the knowledge base is small enough that
TF-IDF retrieval is fast, accurate, and dependency-light.

EXTENDED: Traffic and Energy domains added alongside Hospital.
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
        self.kb_dir = kb_dir
        self.chunks = []   # list of {"text": ..., "source": ...}
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
                if not section.startswith("#"):
                    section = "## " + section
                self.chunks.append({"text": section, "source": fname})

    # ── Build TF-IDF index over all chunks ───────────────────────────────────
    def _build_index(self):
        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    # ── Retrieve top_k most relevant chunks for a query ──────────────────────
    def retrieve(self, query, top_k=2, source_filter=None):
        """
        Parameters
        ----------
        query         : str, natural language description of the situation
        top_k         : int, number of chunks to return
        source_filter : str or None, restrict search to a specific filename

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


# ── Task → knowledge base file mapping ───────────────────────────────────────
TASK_SOURCE_MAP = {
    # HOSPITAL (Sourav — unchanged)
    "bed_allocation":   "bed_allocation_policy.md",
    "er_queue":         "er_queue_policy.md",
    "staff_allocation": "staff_allocation_policy.md",

    # TRAFFIC (Ador)
    "intersection": "intersection_policy.md",
    "pedestrian":   "pedestrian_policy.md",
    "parking":      "parking_policy.md",

    # ENERGY (Ador)
    "solar_scheduling":   "solar_scheduling_policy.md",
    "battery_management": "battery_management_policy.md",
    "grid_interaction":   "grid_interaction_policy.md",
}


def build_query(task, state, action):
    """Turn a (task, state, action) triple into a natural language query for retrieval."""

    # HOSPITAL
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

    # TRAFFIC
    elif task == "intersection":
        return (
            f"cars NS {state['cars_NS']} cars EW {state['cars_EW']} "
            f"wait NS {state['wait_NS']} wait EW {state['wait_EW']} "
            f"phase elapsed {state['phase_elapsed']} action {action} "
            f"signal green switch wait time traffic"
        )
    elif task == "pedestrian":
        return (
            f"pedestrians {state['peds']} vehicles {state['vehs']} "
            f"ped wait {state['ped_wait']} veh wait {state['veh_wait']} "
            f"action {action} crossing safety priority signal"
        )
    elif task == "parking":
        return (
            f"available spots {state['spots']} incoming {state['incoming']} "
            f"queue wait {state['queue_wait']} occupancy {state['occupancy']} "
            f"action {action} parking zone entry capacity congestion"
        )

    # ENERGY
    elif task == "solar_scheduling":
        return (
            f"solar output {state['solar_output']} home consumption {state['home_consumption']} "
            f"battery level {state['battery_level']} time of day {state['time_of_day']} "
            f"action {action} solar use store battery grid schedule"
        )
    elif task == "battery_management":
        return (
            f"battery level {state['battery_level']} solar output {state['solar_output']} "
            f"grid price {state['grid_price']} consumption {state['home_consumption']} "
            f"action {action} charge discharge idle battery storage"
        )
    elif task == "grid_interaction":
        return (
            f"grid price {state['grid_price']} solar surplus {state['solar_surplus']} "
            f"battery level {state['battery_level']} consumption {state['home_consumption']} "
            f"action {action} buy sell grid self sufficient energy"
        )

    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    # Smoke test — hospital
    retriever = PolicyRetriever("knowledge_base")
    print(f"Loaded {len(retriever.chunks)} chunks from knowledge base.\n")

    print("── Hospital test ──")
    q1 = build_query("bed_allocation", {"free_beds": 2, "waiting_patients": 25}, "Transfer")
    r1 = retriever.retrieve(q1, top_k=2, source_filter="bed_allocation_policy.md")
    for r in r1:
        print(f"[{r['source']}] score={r['score']:.3f}: {r['text'][:100]}...\n")

    print("── Traffic test ──")
    q2 = build_query("intersection",
                     {"cars_NS":5,"cars_EW":2,"current_phase":0,
                      "phase_elapsed":3,"wait_NS":7,"wait_EW":1}, "Green NS")
    r2 = retriever.retrieve(q2, top_k=2, source_filter="intersection_policy.md")
    for r in r2:
        print(f"[{r['source']}] score={r['score']:.3f}: {r['text'][:100]}...\n")

    print("── Energy test ──")
    q3 = build_query("battery_management",
                     {"battery_level":8,"solar_output":1,"grid_price":2,"home_consumption":3},
                     "Discharge Battery")
    r3 = retriever.retrieve(q3, top_k=2, source_filter="battery_management_policy.md")
    for r in r3:
        print(f"[{r['source']}] score={r['score']:.3f}: {r['text'][:100]}...\n")
