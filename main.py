import asyncio
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer, util

load_dotenv()

app_state = {}
VECTOR_STORE_ENABLED = os.getenv("VECTOR_STORE_ENABLED", "false").lower() == "true"
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")


def verify_api_key(x_api_key: str = Header()):
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=500, detail="ADMIN_API_KEY not configured on server"
        )
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Loading model...")
    app_state["semantic_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")

    if VECTOR_STORE_ENABLED:
        try:
            from vector_store import TicketVectorStore

            app_state["vector_store"] = TicketVectorStore(
                model=app_state["semantic_model"]
            )
            stats = app_state["vector_store"].get_stats()
            print(f"Vector store loaded ({stats['total_tickets']} tickets indexed)")
        except Exception as e:
            print(f"Warning: Vector store not available: {e}")

    yield

    app_state.clear()


app = FastAPI(
    title="Jira Duplicate Finder",
    version="1.0.0",
    lifespan=lifespan,
)


class SimilarIssueResponse(BaseModel):
    key: str
    link: str
    summary: str
    status: str
    match_score: str


def parse_jira_description(document_node) -> str:
    """Recursively parses Atlassian Document Format (ADF) to plain text."""
    if not isinstance(document_node, dict):
        return str(document_node) if document_node else ""

    text_chunks = ""
    if document_node.get("type") == "text":
        text_chunks += document_node.get("text", "") + " "

    for child_node in document_node.get("content", []):
        text_chunks += parse_jira_description(child_node)

    return text_chunks.strip()


def build_smart_jql_query(issue_id: str, summary: str) -> str:
    """Builds a JQL query using key words from the summary to narrow down candidates."""
    target_project = issue_id.split("-")[0]

    clean_summary = re.sub(r"[^\w\s]", " ", summary).strip()
    meaningful_words = [word for word in clean_summary.split() if len(word) > 4]
    core_keywords = sorted(meaningful_words, key=len, reverse=True)[:3]

    if core_keywords:
        keyword_filters = " AND ".join(f'text ~ "{word}"' for word in core_keywords)
        return f'project = "{target_project}" AND ({keyword_filters}) AND issueKey != {issue_id}'

    return (
        f'project = "{target_project}" AND issueKey != {issue_id} ORDER BY created DESC'
    )


class JiraClient:
    def __init__(self, domain: str, email: str, api_token: str):
        self.base_url = f"https://{domain}/rest/api/3"
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {"Accept": "application/json"}

    def get_issue_details(self, issue_id: str) -> dict:
        url = f"{self.base_url}/issue/{issue_id}"
        try:
            response = requests.get(
                url, headers=self.headers, auth=self.auth, timeout=10
            )

            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Issue {issue_id} not found in Jira.",
                )
            response.raise_for_status()

            fields = response.json().get("fields", {})

            # Parse comments so the query embedding matches how tickets are indexed
            comments = ""
            comment_field = fields.get("comment", {})
            for c in comment_field.get("comments", []):
                text = parse_jira_description(c.get("body", {}))
                if text:
                    comments += f" {text}"

            return {
                "status": fields.get("status"),
                "summary": fields.get("summary", ""),
                "description": fields.get("description", {}),
                "comments": comments.strip(),
            }
        except requests.exceptions.RequestException as error:
            raise HTTPException(status_code=502, detail=f"Jira API error: {str(error)}")

    def fetch_candidate_pool(self, jql: str, max_results: int = 300) -> list:
        url = f"{self.base_url}/search/jql"
        params = {
            "jql": jql,
            "maxResults": max_results,
            "fields": "summary,status,description",
        }
        try:
            response = requests.get(
                url, headers=self.headers, params=params, auth=self.auth, timeout=20
            )
            response.raise_for_status()
            return response.json().get("issues", [])
        except requests.exceptions.RequestException as error:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch candidates from Jira: {str(error)}",
            )


API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
DOMAIN = os.getenv("JIRA_DOMAIN", "")
EMAIL = os.getenv("JIRA_EMAIL", "")

if not EMAIL or not API_TOKEN:
    raise RuntimeError("Missing Jira credentials. Check your .env file.")

jira_client = JiraClient(domain=DOMAIN, email=EMAIL, api_token=API_TOKEN)


def rank_issues_by_similarity(
    target_text: str, candidates: list, max_results: int
) -> list:
    """Ranks candidate issues by cosine similarity to target text."""
    if not candidates:
        return []

    candidate_texts = []
    for issue in candidates:
        summary = issue["fields"].get("summary", "")
        description = parse_jira_description(issue["fields"].get("description", {}))
        candidate_texts.append(f"{summary}. {description}")

    model = app_state["semantic_model"]
    target_vector = model.encode(target_text, convert_to_tensor=True)
    candidate_vectors = model.encode(candidate_texts, convert_to_tensor=True)

    similarity_scores = util.cos_sim(target_vector, candidate_vectors)[0]

    top_k = min(max_results, len(candidate_texts))
    top_results = torch.topk(similarity_scores, k=top_k)

    matches = []
    for score, index in zip(top_results[0], top_results[1]):
        issue = candidates[index]
        pct = round(score.item() * 100, 1)
        status_obj = issue["fields"].get("status") or {}

        matches.append(
            {
                "key": issue["key"],
                "link": f"https://{DOMAIN}/browse/{issue['key']}",
                "summary": issue["fields"].get("summary", ""),
                "status": status_obj.get("name", "Unknown"),
                "match_score": f"{pct}%",
            }
        )

    return matches


@app.get("/api/v1/issues/{issue_id}/similar", response_model=list[SimilarIssueResponse])
def find_similar_issues(
    issue_id: str,
    max_results: int = Query(5, description="Number of results to return"),
    pool_size: int = Query(
        300, description="Candidate pool size (ignored when using vector store)"
    ),
):
    target_issue = jira_client.get_issue_details(issue_id)
    target_summary = target_issue.get("summary", "")

    if not target_summary:
        raise HTTPException(
            status_code=400, detail="This ticket has no summary to compare against."
        )

    target_description = parse_jira_description(target_issue.get("description"))
    target_full_context = f"{target_summary}. {target_description}"

    if "vector_store" in app_state:
        vector_store = app_state["vector_store"]
        project = issue_id.split("-")[0]

        results = vector_store.search(
            query_text=target_full_context,
            top_k=max_results,
            exclude_keys=[issue_id],
            project_filter=project,
        )

        return [
            {
                "key": r["key"],
                "link": f"https://{DOMAIN}/browse/{r['key']}",
                "summary": r["summary"],
                "status": r["status"],
                "match_score": f"{r['score']}%",
            }
            for r in results
        ]

    jql_query = build_smart_jql_query(issue_id, target_summary)

    candidates = jira_client.fetch_candidate_pool(jql_query, max_results=pool_size)

    ranked_results = rank_issues_by_similarity(
        target_text=target_full_context, candidates=candidates, max_results=max_results
    )

    return ranked_results


@app.get("/health")
def health_check():
    status = {
        "status": "healthy",
        "model_loaded": "semantic_model" in app_state,
        "vector_store": False,
        "indexed_tickets": 0,
        "reindex_running": app_state.get("reindex_running", False),
    }
    if "vector_store" in app_state:
        stats = app_state["vector_store"].get_stats()
        status["vector_store"] = True
        status["indexed_tickets"] = stats["total_tickets"]
    return status


class ReindexRequest(BaseModel):
    project: str
    max_tickets: int = 30000
    since: str | None = None
    until: str | None = None
    clear: bool = False


@app.post("/admin/reindex", dependencies=[Depends(verify_api_key)])
async def trigger_reindex(req: ReindexRequest):
    """Triggers a full or incremental re-index of a Jira project."""
    if "vector_store" not in app_state:
        raise HTTPException(status_code=503, detail="Vector store not enabled")

    if app_state.get("reindex_running"):
        return {"status": "already_running"}

    from index_tickets import fetch_all_tickets

    async def _run():
        from datetime import datetime, timezone

        app_state["reindex_running"] = True
        app_state["reindex_status"] = {
            "status": "running",
            "project": req.project,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "new_tickets": 0,
            "skipped_tickets": 0,
            "error": None,
        }
        try:
            tickets = await asyncio.to_thread(
                fetch_all_tickets,
                project=req.project,
                max_tickets=req.max_tickets,
                since=req.since,
                until=req.until,
            )
            store = app_state["vector_store"]
            if req.clear:
                store.delete_all()
            new_count = 0
            skipped_count = 0
            if tickets:
                existing_keys = store.get_existing_keys()
                new_tickets = [t for t in tickets if t["key"] not in existing_keys]
                skipped_count = len(tickets) - len(new_tickets)
                new_count = len(new_tickets)
                if new_tickets:
                    store.add_tickets(new_tickets)
            print(f"Reindex complete: {new_count} new, {skipped_count} skipped")
            app_state["reindex_status"].update(
                status="completed",
                new_tickets=new_count,
                skipped_tickets=skipped_count,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            print(f"Reindex failed: {e}")
            app_state["reindex_status"].update(
                status="failed",
                error=str(e),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        finally:
            app_state["reindex_running"] = False

    asyncio.create_task(_run())
    return {"status": "started", "project": req.project, "clear": req.clear}


@app.get("/admin/reindex/status", dependencies=[Depends(verify_api_key)])
def reindex_status():
    """Returns the status of the last reindex operation."""
    status = app_state.get("reindex_status")
    if not status:
        return {"status": "no reindex has been triggered"}
    return status


@app.post("/webhook/jira", dependencies=[Depends(verify_api_key)])
async def jira_webhook(request: Request):
    """Receives Jira Automation webhooks on ticket creation and posts similar tickets as a comment."""
    body = await request.json()
    issue_key = body.get("issue_key") or body.get("issue", {}).get("key")

    if not issue_key:
        raise HTTPException(status_code=400, detail="Missing issue_key in request body")

    # Get the ticket details
    target_issue = jira_client.get_issue_details(issue_key)
    target_summary = target_issue.get("summary", "")
    if not target_summary:
        return {"status": "skipped", "reason": "No summary on ticket"}

    target_description = parse_jira_description(target_issue.get("description"))
    target_full_context = f"{target_summary}. {target_description}"

    # Find similar tickets
    if "vector_store" not in app_state:
        return {"status": "skipped", "reason": "Vector store not enabled"}

    vector_store = app_state["vector_store"]
    project = issue_key.split("-")[0]
    results = vector_store.search(
        query_text=target_full_context,
        top_k=5,
        exclude_keys=[issue_key],
        project_filter=project,
    )

    if not results:
        return {
            "status": "ok",
            "comment_posted": False,
            "reason": "No similar tickets found",
        }

    # Only comment if the top match is reasonably similar (>60%)
    if results[0]["score"] < 60:
        return {"status": "ok", "comment_posted": False, "reason": "No strong matches"}

    # Build the comment in Atlassian Document Format
    comment_lines = []
    for r in results[:3]:
        comment_lines.append(
            f"• [{r['key']}|https://{DOMAIN}/browse/{r['key']}] — {r['summary']} ({r['score']}% match)"
        )

    comment_text = (
        "*Potential duplicates detected:*\n\n"
        + "\n".join(comment_lines)
        + "\n\n_Jira Duplicate Finder_"
    )

    # Post the comment back to Jira
    comment_url = f"{jira_client.base_url}/issue/{issue_key}/comment"
    comment_body = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": comment_text}],
                }
            ],
        }
    }

    try:
        resp = requests.post(
            comment_url,
            json=comment_body,
            auth=jira_client.auth,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return {"status": "ok", "comment_posted": True, "matches": len(results[:3])}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "comment_posted": False, "error": str(e)}


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(STATIC_DIR / "index.html"))
