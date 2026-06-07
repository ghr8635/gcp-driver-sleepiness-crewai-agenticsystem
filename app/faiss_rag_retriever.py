import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from google import genai


GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")


class PersistentFaissRAGRetriever:
    """
    Persistent FAISS-based RAG store.

    First run:
        KB JSONL -> Vertex embeddings -> FAISS index -> save to disk

    Next runs:
        load FAISS index + docs metadata from disk

    During inference:
        query -> Vertex embedding -> FAISS search -> top-k context

    Optional growth:
        if Gemini creates a new intervention that is semantically novel,
        add it to the vector DB. If DB is full, replace the most semantically
        redundant existing document.
    """

    def __init__(
        self,
        kb_path: str,
        store_dir: str = "data/faiss_vdb",
        index_name: str = "intervention.index",
        docs_name: str = "intervention_docs.json",
        embedding_model: str = "gemini-embedding-001",
        max_size: int = 50,
        novelty_threshold: float = 0.78,
        sleep_between_calls: float = 0.2,
    ):
        self.kb_path = Path(kb_path)
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / index_name
        self.docs_path = self.store_dir / docs_name

        self.embedding_model = embedding_model
        self.max_size = max_size
        self.novelty_threshold = novelty_threshold
        self.sleep_between_calls = sleep_between_calls

        if not GOOGLE_CLOUD_PROJECT:
            raise ValueError("GOOGLE_CLOUD_PROJECT or PROJECT_ID must be set.")

        self.client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
        )

        self.store_dir.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.docs_path.exists():
            self._load_store()
        else:
            self._build_store_from_kb()

    # -----------------------------
    # Loading / saving
    # -----------------------------

    def _load_kb_jsonl(self) -> List[Dict]:
        docs = []
        with self.kb_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))

        if not docs:
            raise ValueError(f"No documents found in {self.kb_path}")

        normalized_docs = []
        for doc in docs:
            normalized_docs.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "content": doc.get("content"),
                "fan_level": int(doc.get("fan_level")),
                "music": doc.get("music"),
                "vibration": doc.get("vibration"),
                "risk_focus": doc.get("risk_focus", "unknown"),
                "modality_focus": doc.get("modality_focus", "unknown"),
                "source": "seed_kb",
                "usage_count": 0,
                "last_similarity": None,
                "created_by": "initial_knowledge_base",
            })

        return normalized_docs

    def _save_store(self):
        faiss.write_index(self.index, str(self.index_path))

        with self.docs_path.open("w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        print(f"Saved FAISS VDB: {self.index_path}")
        print(f"Saved metadata: {self.docs_path}")

    def _load_store(self):
        self.index = faiss.read_index(str(self.index_path))

        with self.docs_path.open("r", encoding="utf-8") as f:
            self.documents = json.load(f)

        print(f"Loaded FAISS VDB from disk: {self.index_path}")
        print(f"Loaded {len(self.documents)} metadata documents.")

    def _build_store_from_kb(self):
        print("No saved FAISS VDB found. Building first-time persistent vector DB...")

        self.documents = self._load_kb_jsonl()
        vectors = []

        total = len(self.documents)
        for i, doc in enumerate(self.documents, start=1):
            print(f"Embedding seed chunk {i}/{total}: {doc['title']}")
            emb = self._embed_text(self._doc_text(doc))
            vectors.append(emb)
            time.sleep(self.sleep_between_calls)

        embeddings = np.vstack(vectors).astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"Persistent FAISS VDB built with {self.index.ntotal} vectors, dimension {dim}.")
        self._save_store()

    # -----------------------------
    # Embedding / text conversion
    # -----------------------------

    def _embed_text(self, text: str) -> np.ndarray:
        last_error = None

        for attempt in range(5):
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                )
                return np.array(response.embeddings[0].values, dtype="float32")
            except Exception as e:
                last_error = e
                wait_time = min(2 ** attempt, 16)
                print(f"Embedding call failed {attempt + 1}/5: {e}")
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        raise last_error

    def _doc_text(self, doc: Dict) -> str:
        return (
            f"Title: {doc['title']}. "
            f"Content: {doc['content']} "
            f"Recommended fan level: {doc['fan_level']}. "
            f"Recommended music: {doc['music']}. "
            f"Recommended vibration: {doc['vibration']}. "
            f"Risk focus: {doc.get('risk_focus', 'unknown')}. "
            f"Modality focus: {doc.get('modality_focus', 'unknown')}."
        )

    def build_query_from_row(self, row: dict) -> str:
        return (
            f"Driver state has overall risk level {row.get('risk_level')} "
            f"with fatigue_camera {row.get('fatigue_camera')}, "
            f"fatigue_steering {row.get('fatigue_steering')}, "
            f"fatigue_lane {row.get('fatigue_lane')}, "
            f"and fatigue score {row.get('fatigue_score')}. "
            f"Vision features include blink rate {row.get('blink_rate')} per minute, "
            f"yawning rate {row.get('yawning_rate')} per minute, "
            f"and PERCLOS {row.get('perclos')} percent. "
            f"Lane features include SDLP {row.get('sdlp')} meters, "
            f"lane keeping ratio {row.get('lane_keeping_ratio')}, "
            f"and lane departure frequency {row.get('lane_departure_frequency')} per minute. "
            f"Steering features include entropy {row.get('steering_entropy')}, "
            f"steering reversal rate {row.get('steering_reversal_rate')} per minute, "
            f"and steering angle variability {row.get('steering_angle_variability')} degrees. "
            f"Find the most relevant in-cabin intervention policy for fan level, music, vibration, and reason."
        )

    # -----------------------------
    # Retrieval
    # -----------------------------

    def retrieve_with_scores(self, row: dict, top_k: int = 3) -> Tuple[str, float]:
        query = self.build_query_from_row(row)

        query_embedding = self._embed_text(query).reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        contexts = []
        best_score = float(scores[0][0]) if len(scores[0]) else 0.0

        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            doc_idx = int(idx)
            doc = self.documents[doc_idx]

            doc["usage_count"] = int(doc.get("usage_count", 0)) + 1
            doc["last_similarity"] = float(score)

            contexts.append(
                f"Retrieved Context {rank}:\n"
                f"Title: {doc['title']}\n"
                f"Similarity score: {float(score):.4f}\n"
                f"Content: {doc['content']}\n"
                f"Recommended fan level: {doc['fan_level']}\n"
                f"Recommended music: {doc['music']}\n"
                f"Recommended vibration: {doc['vibration']}\n"
            )

        # Save usage metadata.
        self._save_store()

        return "\n".join(contexts), best_score

    def retrieve(self, row: dict, top_k: int = 3) -> str:
        context, _ = self.retrieve_with_scores(row, top_k=top_k)
        return context

    # -----------------------------
    # VDB growth
    # -----------------------------

    def _get_all_vectors(self) -> np.ndarray:
        vectors = []
        for i in range(self.index.ntotal):
            vectors.append(self.index.reconstruct(i))
        return np.vstack(vectors).astype("float32")

    def _find_most_redundant_doc_index(self) -> int:
        """
        Replacement policy:
        - compute pairwise semantic similarity between existing vectors
        - find the document that is most similar to another document
        - replace that because it is the most redundant item

        This keeps the vector DB diverse.
        """
        vectors = self._get_all_vectors()

        sim = vectors @ vectors.T

        # Ignore self-similarity.
        np.fill_diagonal(sim, -1.0)

        # Redundancy score = highest similarity to any other doc.
        redundancy_scores = sim.max(axis=1)

        replace_idx = int(np.argmax(redundancy_scores))
        return replace_idx

    def maybe_add_intervention(
        self,
        row: dict,
        parsed: dict,
        llm_output: str,
        best_retrieval_score: float,
    ) -> Dict:
        """
        Add a new intervention to the VDB only if it is semantically novel.

        If best_retrieval_score is high, it means the existing DB already
        has similar knowledge, so we do not add.

        If best_retrieval_score is low, we add the generated intervention.

        If DB is full, we replace the most semantically redundant existing doc.
        """
        if best_retrieval_score >= self.novelty_threshold:
            return {
                "added": False,
                "reason": f"Not novel enough. Best retrieval score={best_retrieval_score:.4f}",
                "best_retrieval_score": best_retrieval_score,
            }

        fan_level = parsed.get("fan_level")
        music = parsed.get("music")
        vibration = parsed.get("vibration")
        reason = parsed.get("reason")

        if fan_level is None or music is None or vibration is None or not reason:
            return {
                "added": False,
                "reason": "LLM output could not be parsed cleanly.",
                "best_retrieval_score": best_retrieval_score,
            }

        new_id = f"gen_{int(time.time())}_{len(self.documents)}"
        new_doc = {
            "id": new_id,
            "title": f"Generated intervention for {row.get('risk_level')} fatigue",
            "content": (
                f"Generated from live/API LLM output. "
                f"Driver risk level was {row.get('risk_level')} with fatigue score {row.get('fatigue_score')}. "
                f"Camera fatigue was {row.get('fatigue_camera')}, steering fatigue was {row.get('fatigue_steering')}, "
                f"and lane fatigue was {row.get('fatigue_lane')}. "
                f"Recommended intervention from LLM: {llm_output}"
            ),
            "fan_level": int(fan_level),
            "music": music,
            "vibration": vibration,
            "risk_focus": row.get("risk_level", "unknown"),
            "modality_focus": "generated_multimodal",
            "source": "llm_generated",
            "usage_count": 0,
            "last_similarity": None,
            "created_by": "vertex_gemini_output",
        }

        new_vector = self._embed_text(self._doc_text(new_doc)).reshape(1, -1).astype("float32")
        faiss.normalize_L2(new_vector)

        if self.index.ntotal < self.max_size:
            self.documents.append(new_doc)
            self.index.add(new_vector)
            self._save_store()

            return {
                "added": True,
                "action": "added",
                "doc_id": new_id,
                "best_retrieval_score": best_retrieval_score,
                "current_size": self.index.ntotal,
            }

        replace_idx = self._find_most_redundant_doc_index()
        replaced_doc = self.documents[replace_idx]

        # Rebuild index with replacement.
        all_vectors = self._get_all_vectors()
        all_vectors[replace_idx] = new_vector[0]

        dim = all_vectors.shape[1]
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(all_vectors.astype("float32"))

        self.index = new_index
        self.documents[replace_idx] = new_doc
        self._save_store()

        return {
            "added": True,
            "action": "replaced_redundant_doc",
            "doc_id": new_id,
            "replaced_doc_id": replaced_doc.get("id"),
            "replaced_doc_title": replaced_doc.get("title"),
            "best_retrieval_score": best_retrieval_score,
            "current_size": self.index.ntotal,
        }


# Backward-compatible alias so existing imports keep working.
FaissVertexRAGRetriever = PersistentFaissRAGRetriever


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    retriever = PersistentFaissRAGRetriever(
        kb_path=str(root / "data/intervention_knowledge_base.jsonl"),
        store_dir=str(root / "data/faiss_vdb"),
        max_size=50,
    )

    sample_row = {
        "risk_level": "high",
        "fatigue_camera": "high",
        "fatigue_steering": "high",
        "fatigue_lane": "medium",
        "fatigue_score": 0.78,
        "blink_rate": 35.0,
        "yawning_rate": 2.5,
        "perclos": 45.0,
        "sdlp": 0.62,
        "lane_keeping_ratio": 0.63,
        "lane_departure_frequency": 1.8,
        "steering_entropy": 4.8,
        "steering_reversal_rate": 9.5,
        "steering_angle_variability": 8.4,
    }

    context, score = retriever.retrieve_with_scores(sample_row, top_k=3)
    print(context)
    print(f"Best retrieval score: {score:.4f}")
