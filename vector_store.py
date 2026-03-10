"""Vector store for Jira ticket embeddings using ChromaDB."""

import os
import platform
from datetime import datetime

import chromadb
import torch
from chromadb.api.types import Metadata
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")


def get_device() -> str:
    """Detect the best available device for inference."""
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    return "cpu"


def get_collection_name(model_name: str) -> str:
    """Generate collection name based on model, replacing special chars."""
    if model_name == "all-MiniLM-L6-v2":
        return "jira_tickets"  # Default collection for backward compatibility
    # Normalize model name: Qwen/Qwen3-Embedding-8B -> qwen3_embedding_8b
    name = model_name.split("/")[-1].lower().replace("-", "_")
    return f"jira_tickets_{name}"


class TicketVectorStore:
    """Manages vector embeddings for Jira tickets."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model: SentenceTransformer | None = None,
    ):
        self.model_name = model_name
        device = get_device()
        print(f"Using device: {device}")
        self.model = model or SentenceTransformer(model_name, device=device)
        self.client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH, settings=Settings(anonymized_telemetry=False)
        )
        collection_name = get_collection_name(model_name)
        print(f"Using collection: {collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _get_max_chars(self) -> int:
        """Get max characters based on model context length. ~4 chars per token."""
        model_lower = self.model_name.lower()
        if "qwen3" in model_lower:
            return 4000  # ~1K tokens, conservative for batching
        elif "nomic" in model_lower:
            return 4000  # ~1K tokens
        return 2000  # MiniLM default

    def _build_text(self, ticket: dict, max_chars: int | None = None) -> str:
        """Concatenates summary, description, and comments into searchable text."""
        if max_chars is None:
            max_chars = self._get_max_chars()
        
        parts = [ticket["summary"]]

        if ticket.get("description"):
            parts.append(ticket["description"])

        if ticket.get("comments"):
            parts.append(f"Comments: {ticket['comments']}")

        text = ". ".join(parts)
        return text[:max_chars] if len(text) > max_chars else text

    def add_tickets(self, tickets: list[dict], batch_size: int = 10):
        """
        Add tickets to the vector store.

        Args:
            tickets: List of dicts with 'key', 'summary', 'description', 'status', 'comments'
            batch_size: Number of tickets to encode at once (default: 10 for large models)
        """
        import time
        
        start_time = time.time()
        
        for i in range(0, len(tickets), batch_size):
            batch = tickets[i : i + batch_size]

            ids = [t["key"] for t in batch]
            texts = [self._build_text(t) for t in batch]
            embeddings = self.model.encode(texts).tolist()

            metadatas: list[Metadata] = [
                {
                    "summary": t["summary"],
                    "status": t.get("status", "Unknown"),
                    "has_comments": bool(t.get("comments")),
                    "indexed_at": datetime.now().isoformat(),
                }
                for t in batch
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
            )

            indexed = min(i + batch_size, len(tickets))
            elapsed = time.time() - start_time
            rate = indexed / elapsed if elapsed > 0 else 0
            remaining = len(tickets) - indexed
            eta_min = remaining / rate / 60 if rate > 0 else 0
            print(f"Indexed {indexed}/{len(tickets)} ({rate:.1f}/sec, ETA: {eta_min:.0f}min)")

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        exclude_keys: list[str] | None = None,
        project_filter: str | None = None,
    ) -> list[dict]:
        """
        Search for similar tickets.

        Args:
            query_text: The text to find similar tickets for
            top_k: Number of results to return
            exclude_keys: Ticket keys to exclude (e.g., the source ticket)
            project_filter: Only return tickets from this project

        Returns:
            List of dicts with 'key', 'summary', 'status', 'score'
        """
        query_embedding = self.model.encode(query_text).tolist()

        fetch_k = top_k + len(exclude_keys or []) + 10

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["metadatas", "distances"],
        )

        exclude_set = set(exclude_keys or [])
        distances = results["distances"] or []
        metadatas = results["metadatas"] or []
        matches = []

        for i, key in enumerate(results["ids"][0]):
            if key in exclude_set:
                continue

            if project_filter and not key.startswith(f"{project_filter}-"):
                continue

            distance = distances[0][i]
            score = 1 - distance

            metadata = metadatas[0][i]
            matches.append(
                {
                    "key": key,
                    "summary": metadata.get("summary", ""),
                    "status": metadata.get("status", "Unknown"),
                    "score": round(score * 100, 1),
                }
            )

            if len(matches) >= top_k:
                break

        return matches

    def get_existing_keys(self) -> set[str]:
        """Returns the set of ticket keys already in the store."""
        result = self.collection.get(include=[])
        return set(result["ids"])

    def get_stats(self) -> dict:
        """Returns statistics about the vector store."""
        return {
            "total_tickets": self.collection.count(),
            "db_path": VECTOR_DB_PATH,
        }

    def delete_all(self):
        """Clears all tickets from the store."""
        collection_name = get_collection_name(self.model_name)
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
