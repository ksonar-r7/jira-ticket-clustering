"""Vector store for Jira ticket embeddings using ChromaDB."""

import os
from datetime import datetime

import chromadb
from chromadb.api.types import Metadata
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")


class TicketVectorStore:
    """Manages vector embeddings for Jira tickets."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model: SentenceTransformer | None = None,
    ):
        self.model = model or SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="jira_tickets", metadata={"hnsw:space": "cosine"}
        )

    def _build_text(self, ticket: dict) -> str:
        """Concatenates summary, description, and comments into searchable text."""
        parts = [ticket["summary"]]

        if ticket.get("description"):
            parts.append(ticket["description"])

        if ticket.get("comments"):
            parts.append(f"Comments: {ticket['comments']}")

        return ". ".join(parts)

    def add_tickets(self, tickets: list[dict], batch_size: int = 100):
        """
        Add tickets to the vector store.

        Args:
            tickets: List of dicts with 'key', 'summary', 'description', 'status', 'comments'
        """
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

            print(f"Indexed {min(i + batch_size, len(tickets))}/{len(tickets)} tickets")

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
        self.client.delete_collection("jira_tickets")
        self.collection = self.client.get_or_create_collection(
            name="jira_tickets", metadata={"hnsw:space": "cosine"}
        )
