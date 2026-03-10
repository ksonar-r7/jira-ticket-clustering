#!/usr/bin/env python3
"""
Index all Jira tickets into the vector store.

Run this script periodically (e.g., daily cron job) to keep the vector store
in sync with Jira. It fetches all tickets and computes embeddings for semantic search.

Usage:
    python index_tickets.py --project SI --max-tickets 10000

Incremental update:
    python index_tickets.py --project SI --since 2026-01-01

Date range:
    python index_tickets.py --project SI --since 2025-01-01 --until 2025-12-31
"""

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from vector_store import TicketVectorStore

load_dotenv()

TICKET_CACHE_DIR = Path(os.getenv("TICKET_CACHE_DIR", "./ticket_cache"))


def get_cache_path(project: str) -> Path:
    """Get the cache file path for a project."""
    return TICKET_CACHE_DIR / f"{project.lower()}_tickets.json"


def load_from_cache(project: str) -> list[dict] | None:
    """Load tickets from cache file if it exists."""
    cache_path = get_cache_path(project)
    if cache_path.exists():
        print(f"Loading tickets from cache: {cache_path}")
        with open(cache_path) as f:
            tickets = json.load(f)
        print(f"  Loaded {len(tickets)} tickets from cache")
        return tickets
    return None


def save_to_cache(project: str, tickets: list[dict]):
    """Save tickets to cache file."""
    TICKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(project)
    with open(cache_path, "w") as f:
        json.dump(tickets, f)
    print(f"  Saved {len(tickets)} tickets to cache: {cache_path}")


def parse_description(doc_node) -> str:
    """Recursively parses Atlassian Document Format to plain text."""
    if not isinstance(doc_node, dict):
        return str(doc_node) if doc_node else ""

    text = ""
    if doc_node.get("type") == "text":
        text += doc_node.get("text", "") + " "

    for child in doc_node.get("content", []):
        text += parse_description(child)

    return text.strip()


def parse_inline_comments(comment_field: dict) -> str:
    """
    Parse comments returned inline from Jira search API.

    The comment field in the search response has the structure:
    {"comments": [...], "maxResults": N, "total": N, "startAt": 0}
    """
    parts = []
    for comment in comment_field.get("comments", []):
        body = comment.get("body", {})
        text = parse_description(body)
        if text:
            parts.append(text)

    return " ".join(parts)


def fetch_all_tickets(
    project: str,
    max_tickets: int = 10000,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    """
    Fetches all tickets from a Jira project, including comments.

    Args:
        project: Jira project key
        max_tickets: Maximum number of tickets to fetch
        since: Only fetch tickets updated on or after this date (YYYY-MM-DD)
        until: Only fetch tickets updated on or before this date (YYYY-MM-DD)
    """
    domain = os.getenv("JIRA_DOMAIN", "")
    email = os.getenv("JIRA_EMAIL", "")
    api_token = os.getenv("JIRA_API_TOKEN", "")

    if not all([domain, email, api_token]):
        raise RuntimeError("Missing Jira credentials in .env file")

    base_url = f"https://{domain}/rest/api/3"
    auth = HTTPBasicAuth(email, api_token)
    headers = {"Accept": "application/json"}

    # Build JQL query
    jql = f'project = "{project}"'
    if since:
        jql += f' AND updated >= "{since}"'
    if until:
        jql += f' AND updated <= "{until}"'
    jql += " ORDER BY created DESC"

    url = f"{base_url}/search/jql"
    all_tickets = []
    next_page_token = None
    page_size = 100

    fields = "summary,description,status,comment"

    print(f"Fetching tickets from project {project}...")
    if since and until:
        print(f"  (tickets updated between {since} and {until})")
    elif since:
        print(f"  (only tickets updated since {since})")
    elif until:
        print(f"  (only tickets updated until {until})")

    while len(all_tickets) < max_tickets:
        params = {
            "jql": jql,
            "maxResults": min(page_size, max_tickets - len(all_tickets)),
            "fields": fields,
        }

        if next_page_token:
            params["nextPageToken"] = next_page_token

        try:
            response = requests.get(
                url, headers=headers, params=params, auth=auth, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            issues = data.get("issues", [])
            if not issues:
                print("  No more issues returned")
                break

            for issue in issues:
                fields_data = issue.get("fields", {})
                status_obj = fields_data.get("status") or {}

                comment_field = fields_data.get("comment", {})
                comments_text = parse_inline_comments(comment_field)

                ticket = {
                    "key": issue["key"],
                    "summary": fields_data.get("summary", ""),
                    "description": parse_description(
                        fields_data.get("description", {})
                    ),
                    "status": status_obj.get("name", "Unknown"),
                    "comments": comments_text,
                }

                all_tickets.append(ticket)

            print(f"  Fetched {len(all_tickets)}/{max_tickets} tickets...")

            is_last = data.get("isLast", True)
            next_page_token = data.get("nextPageToken")

            if is_last or not next_page_token:
                break

        except requests.exceptions.RequestException as error:
            print(f"Error fetching tickets: {error}")
            break

    comments_found = sum(1 for t in all_tickets if t["comments"])
    print(f"  Found comments on {comments_found}/{len(all_tickets)} tickets")

    return all_tickets


def main():
    parser = argparse.ArgumentParser(
        description="Index Jira tickets into the vector store"
    )
    parser.add_argument(
        "--project", "-p", required=True, help="Jira project key to index (e.g., SI)"
    )
    parser.add_argument(
        "--max-tickets",
        "-m",
        type=int,
        default=10000,
        help="Maximum number of tickets to index (default: 10000)",
    )
    parser.add_argument(
        "--since",
        "-s",
        help="Only index tickets updated on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--until",
        "-u",
        help="Only index tickets updated on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing index before indexing"
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model to use (default: all-MiniLM-L6-v2). Examples: Qwen/Qwen3-Embedding-8B, Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load tickets from local cache instead of fetching from Jira",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding (default: 64). Use smaller values for large models.",
    )

    args = parser.parse_args()

    # Initialize vector store
    print(f"\nInitializing vector store with model: {args.model}")
    store = TicketVectorStore(model_name=args.model)

    if args.clear:
        print("Clearing existing index...")
        store.delete_all()

    # Show current stats
    stats = store.get_stats()
    print(f"Current index: {stats['total_tickets']} tickets")

    # Load tickets from cache or fetch from Jira
    if args.use_cache:
        tickets = load_from_cache(args.project)
        if tickets is None:
            print(f"No cache found for project {args.project}. Fetching from Jira...")
            tickets = fetch_all_tickets(
                project=args.project,
                max_tickets=args.max_tickets,
                since=args.since,
                until=args.until,
            )
            if tickets:
                save_to_cache(args.project, tickets)
    else:
        tickets = fetch_all_tickets(
            project=args.project,
            max_tickets=args.max_tickets,
            since=args.since,
            until=args.until,
        )
        if tickets:
            save_to_cache(args.project, tickets)

    if not tickets:
        print("No tickets to index.")
        return

    # Skip tickets already in the store
    existing_keys = store.get_existing_keys()
    new_tickets = [t for t in tickets if t["key"] not in existing_keys]
    skipped = len(tickets) - len(new_tickets)

    if skipped:
        print(f"\nSkipping {skipped} tickets already in the index")

    if not new_tickets:
        print("All fetched tickets are already indexed.")
        return

    print(f"Indexing {len(new_tickets)} new tickets...")
    store.add_tickets(new_tickets, batch_size=args.batch_size)

    # Final stats
    stats = store.get_stats()
    print(f"\nDone. Index now contains {stats['total_tickets']} tickets")
    print(f"Stored at: {stats['db_path']}")


if __name__ == "__main__":
    main()
