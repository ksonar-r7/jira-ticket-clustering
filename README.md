# Jira Duplicate Finder

A semantic search API that finds duplicate Jira tickets using sentence embeddings. Given a ticket ID, it returns the most similar existing tickets ranked by cosine similarity.

Built with FastAPI, ChromaDB, and the `all-MiniLM-L6-v2` sentence-transformer model (384 dimensions). Achieves 93.3% Recall@5 on historic duplicate tickets.

## How It Works

1. Tickets (summary + description + comments) are fetched from Jira and embedded into vectors
2. Vectors are stored in ChromaDB for fast nearest-neighbor search
3. When queried, the input ticket is embedded and compared against the store using cosine similarity
4. The top-K most similar tickets are returned with match scores

If no vector store is available, the API falls back to a JQL-based search with real-time similarity ranking.

## Project Structure

```
main.py                 # FastAPI server (API, webhook, admin endpoints)
vector_store.py         # ChromaDB wrapper for embeddings
index_tickets.py        # CLI tool to fetch and index Jira tickets
evaluate.py             # Evaluation framework for accuracy benchmarking
static/index.html       # Web UI for non-technical users
Dockerfile              # Production container image
docker-compose.yml      # Container orchestration
requirements.txt        # Python dependencies (local dev)
requirements-docker.txt # Python dependencies (Docker, CPU-only PyTorch)
```

## Setup

### Prerequisites

- Python 3.11+
- A Jira Cloud instance with API access

### Environment Variables

Create a `.env` file:

```env
JIRA_DOMAIN=your-org.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_API_TOKEN=your-jira-api-token
ADMIN_API_KEY=a-secret-key-for-admin-endpoints
```

Generate a Jira API token at https://id.atlassian.com/manage-profile/security/api-tokens.

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Index Tickets

```bash
# Full index
python index_tickets.py --project SI --max-tickets 30000

# Incremental (only tickets updated since a date)
python index_tickets.py --project SI --since 2026-01-01

# Date range
python index_tickets.py --project SI --since 2025-01-01 --until 2025-12-31

# Clear and rebuild from scratch
python index_tickets.py --project SI --max-tickets 30000 --clear
```

Already-indexed tickets are automatically skipped. Use `--clear` to force a full rebuild.

#### Start the Server

```bash
VECTOR_STORE_ENABLED=true uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

### Docker Deployment

```bash
docker compose up -d --build
```

This builds the image (with the ML model baked in), starts the API on port 8000, and mounts a named volume for the vector store data.

## API Reference

### Find Similar Tickets

```
GET /api/v1/issues/{issue_id}/similar
```

**Public** — no authentication required.

| Parameter     | Type | Default | Description                                      |
|---------------|------|---------|--------------------------------------------------|
| `issue_id`    | path | —       | Jira issue key (e.g., `SI-12345`)                |
| `max_results` | query| 5       | Number of results to return                      |
| `pool_size`   | query| 300     | JQL candidate pool size (ignored with vector store) |

**Response:**

```json
[
  {
    "key": "SI-11111",
    "link": "https://your-org.atlassian.net/browse/SI-11111",
    "summary": "Login page returns 500 error",
    "status": "Closed",
    "match_score": "87.3%"
  }
]
```

### Health Check

```
GET /health
```

**Public** — no authentication required.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "vector_store": true,
  "indexed_tickets": 28650
}
```

### Trigger Re-index

```
POST /admin/reindex
```

**Authenticated** — requires `X-API-Key` header.

**Request body:**

```json
{
  "project": "SI",
  "max_tickets": 30000,
  "since": "2026-02-01",
  "until": "2026-02-15",
  "clear": false
}
```

Only `project` is required. All other fields are optional.

The endpoint returns immediately and runs indexing in the background. Already-indexed tickets are skipped unless `clear` is set to `true`.

**Example:**

```bash
curl -X POST http://localhost:8000/admin/reindex \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"project": "SI"}'
```

### Jira Webhook

```
POST /webhook/jira
```

**Authenticated** — requires `X-API-Key` header.

Receives a webhook from Jira Automation when a ticket is created. If similar tickets are found (top match >60%), it posts a comment on the new ticket listing potential duplicates.

**Request body:**

```json
{
  "issue_key": "SI-12345"
}
```

To set this up in Jira:
1. Go to Project Settings > Automation
2. Create a rule triggered on "Issue Created"
3. Add a "Send web request" action:
   - URL: `https://your-host:8000/webhook/jira`
   - Method: POST
   - Headers: `X-API-Key: your-secret-key`
   - Body: `{"issue_key": "{{issue.key}}"}`

### Web UI

```
GET /
```

A browser-based interface for non-technical users to search for duplicate tickets. Enter a Jira issue key and view results without needing to call the API directly.

## Evaluation

Run the evaluation framework against historic tickets resolved as "Duplicate":

```bash
python evaluate.py --project SI --test-cases 30 --top-k 5
```

This fetches real duplicate pairs from Jira and measures how often the correct duplicate appears in the top-K results. Results are printed to stdout and optionally saved to a JSON file with `--output results.json`.

## Production Deployment (EC2)

Recommended setup: a single `t3.small` EC2 instance (~$17/mo). The service needs ~1.2 GB RAM (model + vector store + overhead), no GPU.

### 1. Launch the Instance

- AMI: Amazon Linux 2023 or Ubuntu 22.04
- Instance type: `t3.small` (2 vCPU, 2 GB RAM)
- Storage: 20 GB gp3
- Security group: allow inbound on port 8000 from your VPC (or wherever Jira Automation sends webhooks from)

### 2. Install Docker

```bash
# Amazon Linux 2023
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# Install Docker Compose plugin
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

Log out and back in for the group change to take effect.

### 3. Deploy

```bash
git clone <repo-url> /opt/jira-duplicate-finder
cd /opt/jira-duplicate-finder
cp .env.example .env   # edit with your credentials
docker compose up -d --build
```

### 4. Initial Index

```bash
curl -X POST http://localhost:8000/admin/reindex \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ADMIN_API_KEY" \
  -d '{"project": "SI", "max_tickets": 30000}'
```

Monitor progress by polling the health endpoint:

```bash
watch -n5 'curl -s http://localhost:8000/health | python3 -m json.tool'
```

### 5. Nightly Incremental Sync (Cron)

Set up a cron job to index tickets updated in the last 24 hours:

```bash
crontab -e
```

Add:

```cron
0 2 * * * curl -s -X POST http://localhost:8000/admin/reindex -H "Content-Type: application/json" -H "X-API-Key: YOUR_API_KEY" -d '{"project":"SI","since":"'$(date -d yesterday +\%Y-\%m-\%d)'"}'
```

On macOS (for local testing), use `date -v-1d` instead:

```cron
0 2 * * * curl -s -X POST http://localhost:8000/admin/reindex -H "Content-Type: application/json" -H "X-API-Key: YOUR_API_KEY" -d '{"project":"SI","since":"'$(date -v-1d +\%Y-\%m-\%d)'"}'
```

This runs daily at 2 AM. Only new tickets are indexed; existing ones are skipped automatically.
