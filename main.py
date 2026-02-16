import os
import re
from contextlib import asynccontextmanager

import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# We keep the AI model in memory here so it only loads once when the server starts.
app_state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Handles startup and shutdown events for the API."""
    print("Loading semantic model into memory.")
    app_state["semantic_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    print("AI Model loaded!")

    yield  # The API runs while yielded here

    print("Shutting down gracefully...")
    app_state.clear()


app = FastAPI(
    title="Jira Issue Matchmaker API",
    description="Finds semantically related Jira tickets using hybrid AI search.",
    version="0.1.0",
    lifespan=lifespan,
)


class SimilarIssueResponse(BaseModel):
    key: str
    link: str
    summary: str
    status: str
    match_score: str


def parse_jira_description(document_node) -> str:
    """
    Jira stores descriptions in a messy, deeply nested format called Atlassian Document Format (ADF).
    This function recursively digs through that JSON to pull out just the plain text.
    """
    if not isinstance(document_node, dict):
        return str(document_node) if document_node else ""

    text_chunks = ""

    # If we found a text block, grab it
    if document_node.get("type") == "text":
        text_chunks += document_node.get("text", "") + " "

    # Keep digging deeper into the content array
    for child_node in document_node.get("content", []):
        text_chunks += parse_jira_description(child_node)

    return text_chunks.strip()


def build_smart_jql_query(issue_id: str, summary: str) -> str:
    """
    Instead of searching every ticket in history, we find the longest, most unique
    words in the target issue's summary and use them to narrow down our candidate pool.
    """
    target_project = issue_id.split("-")[0]

    # Strip punctuation and get words longer than 4 characters
    clean_summary = re.sub(r"[^\w\s]", " ", summary).strip()
    meaningful_words = [word for word in clean_summary.split() if len(word) > 4]

    # Grab the top 3 longest words, assuming they are the most technically unique
    core_keywords = sorted(meaningful_words, key=len, reverse=True)[:3]

    if core_keywords:
        # Force Jira to find tickets containing ALL of these specific keywords
        keyword_filters = " AND ".join(f'text ~ "{word}"' for word in core_keywords)
        return f'project = "{target_project}" AND ({keyword_filters}) AND issueKey != {issue_id}'

    # Fallback: If the summary is too short, just grab the most recent tickets
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
                    detail=f"Hold up—we couldn't find {issue_id} in Jira.",
                )
            response.raise_for_status()

            fields = response.json().get("fields", {})
            return {
                "status": fields.get("status"),
                "summary": fields.get("summary", ""),
                "description": fields.get("description", {}),
            }
        except requests.exceptions.RequestException as error:
            raise HTTPException(
                status_code=502, detail=f"Jira API is unhappy: {str(error)}"
            )

    def fetch_candidate_pool(self, jql: str, max_results: int = 300) -> list:
        url = f"{self.base_url}/search/jql"
        payload = {
            "jql": jql,
            "maxResults": max_results,
            "fields": ["summary", "status", "description"],
        }
        try:
            response = requests.post(
                url, headers=self.headers, json=payload, auth=self.auth, timeout=20
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
    raise RuntimeError("Missing Jira credentials! Please check your .env file.")

jira_client = JiraClient(domain=DOMAIN, email=EMAIL, api_token=API_TOKEN)


def rank_issues_by_similarity(
    target_text: str, candidates: list, max_results: int
) -> list:
    """
    Takes a target text and a list of candidate Jira issues, converts them all into
    AI vector embeddings, and calculates how closely their meanings match.
    """
    if not candidates:
        return []

    candidate_texts = []
    for issue in candidates:
        summary = issue["fields"].get("summary", "")
        description = parse_jira_description(issue["fields"].get("description", {}))
        candidate_texts.append(f"{summary}. {description}")

    ai_model = app_state["semantic_model"]
    target_vector = ai_model.encode(target_text, convert_to_tensor=True)
    candidate_vectors = ai_model.encode(candidate_texts, convert_to_tensor=True)

    similarity_scores = util.cos_sim(target_vector, candidate_vectors)[0]

    top_k = min(max_results, len(candidate_texts))
    top_results = torch.topk(similarity_scores, k=top_k)

    winning_issues = []
    for score, index in zip(top_results[0], top_results[1]):
        matched_issue = candidates[index]
        match_percentage = round(score.item() * 100, 1)

        status_obj = matched_issue["fields"].get("status") or {}
        status_name = status_obj.get("name", "Unknown")

        winning_issues.append(
            {
                "key": matched_issue["key"],
                "link": f"https://{DOMAIN}/browse/{matched_issue['key']}",
                "summary": matched_issue["fields"].get("summary", ""),
                "status": status_name,
                "match_score": f"{match_percentage}%",
            }
        )

    return winning_issues


@app.get("/api/v1/issues/{issue_id}/similar", response_model=list[SimilarIssueResponse])
def find_similar_issues(
    issue_id: str,
    max_results: int = Query(5, description="How many top matches do you want back?"),
    pool_size: int = Query(
        300, description="How many historical tickets should we evaluate?"
    ),
):
    """
    It fetches the target issue, gathers a pool of likely candidates,
    and asks the AI to rank them by semantic meaning.
    """
    target_issue = jira_client.get_issue_details(issue_id)
    target_summary = target_issue.get("summary", "")

    if not target_summary:
        raise HTTPException(
            status_code=400, detail="This ticket has no summary to compare against."
        )

    target_description = parse_jira_description(target_issue.get("description"))
    target_full_context = f"{target_summary}. {target_description}"

    jql_query = build_smart_jql_query(issue_id, target_summary)

    candidates = jira_client.fetch_candidate_pool(jql_query, max_results=pool_size)

    ranked_results = rank_issues_by_similarity(
        target_text=target_full_context, candidates=candidates, max_results=max_results
    )

    return ranked_results
