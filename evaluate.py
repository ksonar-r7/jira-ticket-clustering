#!/usr/bin/env python3
"""
Evaluation script for Jira ticket similarity model.

This script benchmarks the accuracy of the semantic similarity model by testing
against ground truth data: tickets that have been manually linked as duplicates in Jira.

Metrics calculated:
- Recall@K: What % of known duplicates appear in the top K results?
- Mean Reciprocal Rank (MRR): On average, what position is the correct match found?
- Hit Rate: What % of queries found at least one correct match?

Usage:
    python evaluate.py --project PROJ --limit 100 --top-k 5
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# Check if vector store is available
USE_VECTOR_STORE = os.getenv("VECTOR_STORE_ENABLED", "false").lower() == "true"


@dataclass
class EvaluationResult:
    """Stores results for a single evaluation case."""

    source_key: str
    expected_duplicates: list[str]
    predicted_keys: list[str]
    predicted_scores: list[float]
    found_at_rank: int | None  # None if not found in top-K
    hit: bool


class JiraEvaluationClient:
    """Handles Jira API calls for fetching evaluation data."""

    def __init__(self, domain: str, email: str, api_token: str):
        self.base_url = f"https://{domain}/rest/api/3"
        self.domain = domain
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {"Accept": "application/json"}

    def fetch_duplicate_pairs(self, project: str, limit: int = 100) -> list[dict]:
        """
        Fetches issues that have duplicate links in Jira.
        Returns pairs of (source_issue, duplicate_issues).
        """
        # JQL to find issues with duplicate links
        jql = (
            f'project = "{project}" AND issueFunction in linkedIssuesOf('
            f'"project = {project}", "is duplicated by") ORDER BY created DESC'
        )

        # Fallback JQL if issueFunction is not available
        fallback_jql = (
            f'project = "{project}" AND status = Closed ORDER BY resolved DESC'
        )

        url = f"{self.base_url}/search/jql"

        try:
            params = {
                "jql": jql,
                "maxResults": limit,
                "fields": "summary,description,comment,issuelinks,status,resolution",
            }
            response = requests.get(
                url, headers=self.headers, params=params, auth=self.auth, timeout=30
            )

            if response.status_code == 400:
                # issueFunction might not be available, try fallback
                print("Note: Using fallback query (issueFunction not available)")
                params["jql"] = fallback_jql
                response = requests.get(
                    url, headers=self.headers, params=params, auth=self.auth, timeout=30
                )

            response.raise_for_status()
            return response.json().get("issues", [])

        except requests.exceptions.RequestException as error:
            print(f"Error fetching issues: {error}")
            return []

    def fetch_issues_with_any_links(self, project: str, limit: int = 100) -> list[dict]:
        """
        Fetches closed issues that have ANY issue links (relates to, blocks, etc).
        Fallback when no explicit duplicate links exist.
        """
        jql = f'project = "{project}" AND status = Closed AND issueLink IS NOT EMPTY ORDER BY resolved DESC'

        url = f"{self.base_url}/search/jql"
        params = {
            "jql": jql,
            "maxResults": limit,
            "fields": "summary,description,comment,issuelinks,status,resolution",
        }

        try:
            response = requests.get(
                url, headers=self.headers, params=params, auth=self.auth, timeout=30
            )
            response.raise_for_status()
            return response.json().get("issues", [])
        except requests.exceptions.RequestException as error:
            print(f"Error fetching linked issues: {error}")
            return []

    def fetch_duplicate_resolution_issues(
        self, project: str, limit: int = 100
    ) -> list[dict]:
        """
        Fetches issues that were resolved/closed as 'Duplicate'.
        These should have a parent ticket they duplicate.
        """
        jql = f'project = "{project}" AND resolution = Duplicate ORDER BY resolved DESC'

        url = f"{self.base_url}/search/jql"
        params = {
            "jql": jql,
            "maxResults": limit,
            "fields": "summary,description,comment,issuelinks,status,resolution",
        }

        try:
            response = requests.get(
                url, headers=self.headers, params=params, auth=self.auth, timeout=30
            )
            response.raise_for_status()
            return response.json().get("issues", [])
        except requests.exceptions.RequestException as error:
            print(f"Error fetching duplicate-resolved issues: {error}")
            return []

    def extract_duplicate_links(self, issue: dict) -> list[str]:
        """Extracts duplicate issue keys from an issue's links."""
        duplicates = []
        links = issue.get("fields", {}).get("issuelinks", [])

        for link in links:
            link_type = link.get("type", {}).get("name", "").lower()

            # Check various duplicate link type names
            if "duplicate" in link_type:
                # Outward link: this issue duplicates another
                if "outwardIssue" in link:
                    duplicates.append(link["outwardIssue"]["key"])
                # Inward link: another issue duplicates this one
                if "inwardIssue" in link:
                    duplicates.append(link["inwardIssue"]["key"])

        return duplicates

    def extract_any_links(self, issue: dict) -> list[str]:
        """Extracts ALL linked issue keys (relates to, blocks, duplicates, etc)."""
        linked = []
        links = issue.get("fields", {}).get("issuelinks", [])

        for link in links:
            if "outwardIssue" in link:
                linked.append(link["outwardIssue"]["key"])
            if "inwardIssue" in link:
                linked.append(link["inwardIssue"]["key"])

        return linked

    def get_issue_text(self, issue: dict) -> str:
        """Extracts searchable text from an issue (matches vector store indexing)."""
        fields = issue.get("fields", {})
        summary = fields.get("summary", "")
        description = self._parse_description(fields.get("description", {}))

        # Include comments to match how tickets are indexed in the vector store
        comments = ""
        comment_field = fields.get("comment", {})
        for c in comment_field.get("comments", []):
            text = self._parse_description(c.get("body", {}))
            if text:
                comments += f" {text}"
        comments = comments.strip()

        parts = [summary, description]
        if comments:
            parts.append(f"Comments: {comments}")
        return ". ".join(p for p in parts if p)

    def _parse_description(self, doc_node) -> str:
        """Recursively parses Atlassian Document Format to plain text."""
        if not isinstance(doc_node, dict):
            return str(doc_node) if doc_node else ""

        text = ""
        if doc_node.get("type") == "text":
            text += doc_node.get("text", "") + " "

        for child in doc_node.get("content", []):
            text += self._parse_description(child)

        return text.strip()

    def fetch_candidate_pool(
        self,
        project: str,
        exclude_key: str,
        limit: int = 300,
        must_include: list[str] | None = None,
    ) -> list[dict]:
        """Fetches a pool of candidate issues for similarity comparison."""
        jql = (
            f'project = "{project}" AND issueKey != {exclude_key} ORDER BY created DESC'
        )

        url = f"{self.base_url}/search/jql"
        params = {
            "jql": jql,
            "maxResults": limit,
            "fields": "summary,description,status",
        }

        try:
            response = requests.get(
                url, headers=self.headers, params=params, auth=self.auth, timeout=30
            )
            response.raise_for_status()
            candidates = response.json().get("issues", [])

            # Ensure expected tickets are in the pool (for fair evaluation)
            if must_include:
                existing_keys = {c["key"] for c in candidates}
                missing = [k for k in must_include if k not in existing_keys]

                if missing:
                    # Fetch the missing expected tickets directly
                    missing_jql = f"key in ({','.join(missing)})"
                    params["jql"] = missing_jql
                    params["maxResults"] = len(missing)
                    resp = requests.get(
                        url,
                        headers=self.headers,
                        params=params,
                        auth=self.auth,
                        timeout=30,
                    )
                    if resp.ok:
                        candidates.extend(resp.json().get("issues", []))

            return candidates
        except requests.exceptions.RequestException as error:
            print(f"Error fetching candidates: {error}")
            return []


class ApiEvaluator:
    """Evaluates by calling the running API (e.g. inside Docker)."""

    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

        # Verify the API is reachable
        try:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            resp = requests.get(f"{self.api_url}/health", headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(f"Connected to API at {self.api_url}")
            print(f"  Status: {data.get('status')}")
            print(f"  Indexed tickets: {data.get('indexed_tickets', 'N/A')}")
        except Exception as e:
            print(f"Error: Cannot reach API at {self.api_url}: {e}")
            sys.exit(1)

    def rank_candidates(
        self,
        target_text: str,
        candidates: list[dict],
        jira_client: "JiraEvaluationClient",
        exclude_key: str | None = None,
        project: str | None = None,
    ) -> list[tuple[str, float]]:
        """Calls the API to get similar tickets for the given issue key."""
        if not exclude_key:
            return []

        url = f"{self.api_url}/api/v1/issues/{exclude_key}/similar"
        params = {"max_results": 100}
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            results = resp.json()
            return [
                (r["key"], float(r["match_score"].rstrip("%")) / 100.0)
                for r in results
            ]
        except Exception as e:
            print(f"    API error for {exclude_key}: {e}")
            return []


class SimilarityEvaluator:
    """Evaluates semantic similarity model against ground truth."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", use_vector_store: bool = False
    ):
        self.use_vector_store = use_vector_store
        self.vector_store = None
        self.model: SentenceTransformer | None = None

        if use_vector_store:
            try:
                from vector_store import TicketVectorStore

                print("Loading vector store...")
                self.vector_store = TicketVectorStore()
                stats = self.vector_store.get_stats()
                print(f"Vector store loaded ({stats['total_tickets']} tickets indexed)")
            except Exception as e:
                print(f"Warning: Vector store not available: {e}")
                self.use_vector_store = False

        if not self.use_vector_store:
            print(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            print("Model loaded.")

    def rank_candidates(
        self,
        target_text: str,
        candidates: list[dict],
        jira_client: JiraEvaluationClient,
        exclude_key: str | None = None,
        project: str | None = None,
    ) -> list[tuple[str, float]]:
        """Ranks candidates by semantic similarity to target text."""

        # Use vector store if available (searches ALL indexed tickets)
        if self.use_vector_store and self.vector_store:
            results = self.vector_store.search(
                query_text=target_text,
                top_k=100,  # Get more results for evaluation
                exclude_keys=[exclude_key] if exclude_key else None,
                project_filter=project or None,
            )
            return [(r["key"], r["score"] / 100.0) for r in results]

        # Fallback: compute embeddings on the fly
        if not candidates:
            return []

        candidate_texts = [jira_client.get_issue_text(c) for c in candidates]
        candidate_keys = [c["key"] for c in candidates]

        target_vector = self.model.encode(target_text, convert_to_tensor=True)
        candidate_vectors = self.model.encode(candidate_texts, convert_to_tensor=True)

        scores = util.cos_sim(target_vector, candidate_vectors)[0]

        # Sort by score descending
        ranked = sorted(
            zip(candidate_keys, scores.tolist()), key=lambda x: x[1], reverse=True
        )

        return ranked


def run_evaluation(
    project: str,
    limit: int = 100,
    top_k: int = 5,
    pool_size: int = 300,
    test_file: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Runs the full evaluation pipeline.

    Args:
        project: Jira project key to evaluate
        limit: Maximum number of test cases to evaluate
        top_k: Number of top results to consider for Recall@K
        pool_size: Number of candidate issues to search through
        test_file: Optional JSON file with manual test cases
        api_url: If set, evaluate against a running API instead of local model
        api_key: API key for the running API (if required)

    Returns:
        Dictionary with evaluation metrics and detailed results
    """
    # Initialize clients
    domain = os.getenv("JIRA_DOMAIN", "")
    email = os.getenv("JIRA_EMAIL", "")
    api_token = os.getenv("JIRA_API_TOKEN", "")

    if not all([domain, email, api_token]):
        print("Error: Missing Jira credentials in .env file")
        sys.exit(1)

    jira = JiraEvaluationClient(domain, email, api_token)

    if api_url:
        evaluator = ApiEvaluator(api_url, api_key=api_key or "")
    else:
        evaluator = SimilarityEvaluator(use_vector_store=USE_VECTOR_STORE)

    test_cases = []

    # Option 1: Load from manual test file
    if test_file:
        print(f"\nLoading test cases from {test_file}...")
        try:
            with open(test_file) as f:
                manual_cases = json.load(f)
            # Fetch issue pool once (not per test case)
            issues = jira.fetch_candidate_pool(
                project, exclude_key="NONE-0", limit=1000
            )
            issue_map = {i["key"]: i for i in issues}
            for case in manual_cases:
                if case["source"] in issue_map:
                    test_cases.append(
                        {
                            "issue": issue_map[case["source"]],
                            "duplicates": case["expected"],
                            "source": "manual_file",
                        }
                    )
            print(f"Loaded {len(test_cases)} test cases from file")
        except Exception as e:
            print(f"Error loading test file: {e}")
            return {"error": f"Failed to load test file: {e}"}

    # Option 2: Try tickets resolved as "Duplicate"
    if not test_cases:
        print(f"\nLooking for tickets resolved as 'Duplicate' in project {project}...")
        dup_resolved = jira.fetch_duplicate_resolution_issues(project, limit=limit * 2)
        for issue in dup_resolved:
            linked = jira.extract_duplicate_links(issue) or jira.extract_any_links(
                issue
            )
            if linked:
                test_cases.append(
                    {
                        "issue": issue,
                        "duplicates": linked,
                        "source": "duplicate_resolution",
                    }
                )
            if len(test_cases) >= limit:
                break
        if test_cases:
            print(f"Found {len(test_cases)} tickets resolved as Duplicate")

    # Option 3: Try explicit duplicate links
    if not test_cases:
        print(f"\nLooking for explicit duplicate links in project {project}...")
        issues = jira.fetch_duplicate_pairs(project, limit=limit * 2)
        for issue in issues:
            duplicates = jira.extract_duplicate_links(issue)
            if duplicates:
                test_cases.append(
                    {
                        "issue": issue,
                        "duplicates": duplicates,
                        "source": "duplicate_link",
                    }
                )
            if len(test_cases) >= limit:
                break
        if test_cases:
            print(f"Found {len(test_cases)} issues with duplicate links")

    # Option 4: Fallback to ANY linked issues (less ideal but still useful)
    if not test_cases:
        print("\nNo duplicate data found. Falling back to ANY linked issues...")
        print("(This tests if the model finds related tickets, not exact duplicates)")
        issues = jira.fetch_issues_with_any_links(project, limit=limit * 2)
        for issue in issues:
            linked = jira.extract_any_links(issue)
            if linked:
                test_cases.append(
                    {"issue": issue, "duplicates": linked, "source": "any_link"}
                )
            if len(test_cases) >= limit:
                break
        if test_cases:
            print(f"Found {len(test_cases)} issues with linked tickets")

    if not test_cases:
        print(f"\nNo linked issues found in project {project}")
        print("\nOptions:")
        print("  1. Create a manual test file (see --help for format)")
        print("  2. Manually link some duplicate tickets in Jira")
        print("  3. Try a different project with more linked tickets")
        return {"error": "No test cases found"}

    # Filter out test cases where ALL expected duplicates are cross-project
    # (those projects aren't indexed, so they can never match)
    filtered_cases = []
    skipped_cross_project = 0
    for case in test_cases:
        issue_project = case["issue"]["key"].split("-")[0]
        same_project_dups = [
            d for d in case["duplicates"] if d.startswith(f"{issue_project}-")
        ]
        if same_project_dups:
            case["duplicates"] = same_project_dups
            filtered_cases.append(case)
        else:
            skipped_cross_project += 1

    if skipped_cross_project:
        print(
            f"Skipped {skipped_cross_project} test cases"
            " with only cross-project duplicates (not indexed)"
        )
    test_cases = filtered_cases

    if not test_cases:
        return {"error": "No same-project test cases found after filtering"}

    # Determine ground truth type for reporting
    ground_truth_source = test_cases[0].get("source", "unknown")
    print(f"\nGround truth source: {ground_truth_source}")
    print(f"   Test cases: {len(test_cases)}")

    # Run evaluation
    results: list[EvaluationResult] = []
    reciprocal_ranks = []

    for i, case in enumerate(test_cases, 1):
        issue = case["issue"]
        expected_duplicates = case["duplicates"]
        issue_key = issue["key"]

        print(
            f"[{i}/{len(test_cases)}] Evaluating {issue_key} "
            f"(expects: {', '.join(expected_duplicates[:3])}{'...' if len(expected_duplicates) > 3 else ''})"
        )

        # Get target text
        target_text = jira.get_issue_text(issue)

        # Fetch candidates
        project_key = issue_key.split("-")[0]

        # Check if any expected duplicates are cross-project
        has_cross_project = any(
            not d.startswith(f"{project_key}-") for d in expected_duplicates
        )

        # Use API evaluator or vector store if available, otherwise fetch candidates
        if isinstance(evaluator, ApiEvaluator):
            # API handles project filtering and ticket exclusion internally
            ranked = evaluator.rank_candidates(
                target_text, [], jira, exclude_key=issue_key, project=project_key
            )
        elif evaluator.use_vector_store:
            # Vector store searches all indexed tickets - no need to fetch candidates
            # If cross-project duplicates exist, don't filter by project
            ranked = evaluator.rank_candidates(
                target_text,
                [],
                jira,
                exclude_key=issue_key,
                project=None if has_cross_project else project_key,
            )
        else:
            # Fallback: fetch candidate pool (ensure expected are included for fair eval)
            candidates = jira.fetch_candidate_pool(
                project_key, issue_key, pool_size, must_include=expected_duplicates
            )
            ranked = evaluator.rank_candidates(target_text, candidates, jira)

        predicted_keys = [key for key, _ in ranked[:top_k]]
        predicted_scores = [score for _, score in ranked[:top_k]]

        # Check if any expected duplicate is in top-K
        found_at_rank = None
        for rank, (key, _) in enumerate(ranked, 1):
            if key in expected_duplicates:
                found_at_rank = rank
                break

        hit = found_at_rank is not None and found_at_rank <= top_k

        # Calculate reciprocal rank
        if found_at_rank:
            reciprocal_ranks.append(1.0 / found_at_rank)
        else:
            reciprocal_ranks.append(0.0)

        results.append(
            EvaluationResult(
                source_key=issue_key,
                expected_duplicates=expected_duplicates,
                predicted_keys=predicted_keys,
                predicted_scores=predicted_scores,
                found_at_rank=found_at_rank,
                hit=hit,
            )
        )

        if hit:
            print(f"    Found at rank {found_at_rank}")
        else:
            print(f"    Not found in top {top_k}")

    # Calculate metrics
    total = len(results)
    hits = sum(1 for r in results if r.hit)
    recall_at_k = (hits / total) * 100 if total > 0 else 0
    mrr = (sum(reciprocal_ranks) / total) * 100 if total > 0 else 0

    # Recall at different K values
    recall_at_1 = (
        sum(1 for r in results if r.found_at_rank == 1) / total * 100
        if total > 0
        else 0
    )
    recall_at_3 = (
        sum(1 for r in results if r.found_at_rank and r.found_at_rank <= 3)
        / total
        * 100
        if total > 0
        else 0
    )
    recall_at_10 = (
        sum(1 for r in results if r.found_at_rank and r.found_at_rank <= 10)
        / total
        * 100
        if total > 0
        else 0
    )

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "project": project,
        "config": {
            "test_cases": total,
            "top_k": top_k,
            "pool_size": pool_size,
            "model": "all-MiniLM-L6-v2",
            "ground_truth_source": ground_truth_source,
        },
        "metrics": {
            f"recall@{top_k}": f"{recall_at_k:.1f}%",
            "recall@1": f"{recall_at_1:.1f}%",
            "recall@3": f"{recall_at_3:.1f}%",
            "recall@10": f"{recall_at_10:.1f}%",
            "mrr": f"{mrr:.1f}%",
        },
        "summary": {
            "total_evaluated": total,
            "correct_in_top_k": hits,
            "missed": total - hits,
        },
        "detailed_results": [
            {
                "source": r.source_key,
                "expected": r.expected_duplicates,
                "predicted_top_k": r.predicted_keys,
                "found_at_rank": r.found_at_rank,
                "hit": r.hit,
            }
            for r in results
        ],
    }

    return report


def print_report(report: dict):
    """Prints a formatted evaluation report."""
    if "error" in report:
        print(f"\nError: {report['error']}")
        return

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Project: {report['project']}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Model: {report['config']['model']}")
    print(f"Test Cases: {report['config']['test_cases']}")
    print(f"Candidate Pool Size: {report['config']['pool_size']}")

    print("\nMETRICS:")
    print("-" * 40)
    for metric, value in report["metrics"].items():
        print(f"  {metric:20s}: {value}")

    print("\nSUMMARY:")
    print("-" * 40)
    summary = report["summary"]
    print(f"  Total Evaluated:     {summary['total_evaluated']}")
    print(f"  Found in Top-K:      {summary['correct_in_top_k']}")
    print(f"  Missed:              {summary['missed']}")

    print("\nSAMPLE RESULTS:")
    print("-" * 40)
    for result in report["detailed_results"][:5]:
        status = "+" if result["hit"] else "-"
        rank_info = (
            f"rank {result['found_at_rank']}"
            if result["found_at_rank"]
            else "not found"
        )
        print(
            f"  {status} {result['source']} -> expected {result['expected']} ({rank_info})"
        )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Jira ticket similarity model accuracy"
    )
    parser.add_argument(
        "--project",
        "-p",
        required=True,
        help="Jira project key to evaluate (e.g., PROJ)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=50,
        help="Maximum number of test cases (default: 50)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Top K results to consider (default: 5)",
    )
    parser.add_argument(
        "--pool-size",
        "-s",
        type=int,
        default=300,
        help="Candidate pool size (default: 300)",
    )
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument(
        "--test-file",
        "-f",
        help='JSON file with manual test cases. Format: [{"source": "PROJ-123", "expected": ["PROJ-456", "PROJ-789"]}]',
    )
    parser.add_argument(
        "--api-url",
        help="Evaluate against a running API (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the running API (if admin endpoints are needed)",
    )

    args = parser.parse_args()

    report = run_evaluation(
        project=args.project,
        limit=args.limit,
        top_k=args.top_k,
        pool_size=args.pool_size,
        test_file=args.test_file,
        api_url=args.api_url,
        api_key=args.api_key,
    )

    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
