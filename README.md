# AI War Room Backend

Backend system for automated AI industry intelligence gathering and strategic analysis.

## Architecture

- **Port 8000**: Main backend (FastAPI, LightRAG, content cleaning, API endpoints)
- **Port 8001**: N8N workflow communication server (inter-workflow triggers, notifications)
- **Database**: Supabase (articles, sources, analysis, categories)
- **N8N Workflows**: 12 automated workflows for content collection and processing
- **LightRAG**: Strategic analysis and knowledge graph generation

## Quick Start

### Prerequisites
- Python 3.8+
- N8N instance (cloud or self-hosted)
- Supabase project
- API keys: Gemini, OpenRouter

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd ai-war-room-backend
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Configuration

Create `.env` file: Fill in your .env file using `.env.example` as a template

### Run Servers

**Terminal 1 - Main Backend:**
```bash
cd app
python run.py
# Runs on http://localhost:8000
```

**Terminal 2 - Notification Server:**
```bash
python n8n_notifications.py
# Runs on http://localhost:8001
```

## API Endpoints

### Frontend Data
- `GET /api/dashboard` - Dashboard overview data
- `GET /api/articles` - Filtered articles list
- `GET /api/articles/{id}` - Single article details
- `GET /api/categories` - Strategic categories

### Workflow Triggers
- `POST /api/trigger/manual-pipeline` - Manual full pipeline trigger
- `POST /api/workflows/trigger/summary-generation` - Trigger WF-07
- `POST /api/workflows/trigger/content-filtering` - Trigger WF-08
- `POST /api/workflows/trigger/strategic-analysis` - Trigger WF-09

### System Health
- `GET /api/pipeline/status/{session_id}` - Pipeline progress
- `GET /api/workflows/health` - System health check

### Content Processing
- `POST /api/processing/analyze-batch` - Strategic analysis (LightRAG)
- `GET /api/processing/status` - Processing status

## N8N Workflows

### Data Collection (Daily Automated)
- **WF-01**: Source Discovery (06:00)
- **WF-02**: RSS Content Collection (06:30) 
- **WF-03**: Strategic Categories Update (07:00)
- **WF-04**: Content Validation (07:30)

### Processing Pipeline
- **WF-12**: Orchestration (08:00) - Coordinates all processing
- **WF-07**: Summary Generation (Gemini/OpenRouter)
- **WF-08**: Content Filtering & Relevance Scoring
- **WF-09**: Strategic Analysis Trigger (→ LightRAG)

### Inter-Workflow Communication
Workflows communicate via Port 8001 notification server:
```
WF-07 → Port 8001 → WF-08 → Port 8001 → WF-09 → Port 8000 (LightRAG)
```

## Content Processing Flow

1. **Collection**: RSS feeds → `articles` table (status: `collected`)
2. **Summarization**: Gemini generates summaries → `article_summaries` table
3. **Filtering**: AI relevance scoring → status: `approved`/`rejected`
4. **Analysis**: LightRAG strategic analysis → `strategic_analysis` table (status: `analyzed`)

## Database Schema

### Key Tables
- `articles` - Raw collected articles
- `article_summaries` - AI-generated summaries  
- `strategic_analysis` - LightRAG analysis results
- `strategic_categories` - Analysis categories
- `rss_sources` - RSS feed sources
- `pipeline_runs` - Processing session tracking

## Content Cleaning

Automatic removal of unwanted prefixes:
- arXiv paper identifiers (`arXiv:2504.05801v2 Announce Type:`)
- Academic formatting artifacts
- Applied at API layer before frontend delivery

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `N8N_BASE_URL` | N8N instance URL | `https://workspace.app.n8n.cloud` |
| `BACKEND_BASE_URL` | Main backend URL | `http://localhost:8000` |
| `NOTIFICATION_SERVER_URL` | Communication server | `http://localhost:8001` |
| `SUPABASE_URL` | Supabase project URL | Required |
| `SUPABASE_KEY` | Supabase anon key | Required |
| `GOOGLE_API_KEY` | Gemini API key | Required |
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |

## Troubleshooting

### Common Issues

**"Insufficient articles for processing"**
- WF-12 needs 5+ articles with status='collected' from last 24 hours
- Check RSS sources are active: `SELECT * FROM rss_sources WHERE is_active = true`

**"Summary generation failed"**
- Duplicate constraint on article_summaries table
- Use `INSERT ... ON CONFLICT` in WF-07

**"No articles approved"**  
- Articles failing relevance scoring (< 6.0)
- Check word_count (minimum 100) and source reliability_score

**LightRAG never gets called**
- Pipeline failing before WF-09
- Check WF-07 and WF-08 execution logs

### Health Checks

```bash
# Backend health
curl http://localhost:8000/api/workflows/health

# Database article counts
curl http://localhost:8000/api/dashboard

# Pipeline status
curl http://localhost:8000/api/pipeline/status/your_session_id
```

### Logs

- **Main Backend**: Console output from `python run.py`
- **Notification Server**: Console output from `python test_n8n_notifications.py`
- **N8N Workflows**: N8N execution logs (per workflow)
- **LightRAG**: Stored in `./lightrag_storage/`

## Development

### Project Structure
```
ai-war-room-backend/
├── app/
│   ├── main.py              # FastAPI app, API endpoints
│   ├── run.py               # Development server
│   └── services/
│       ├── gemini_service.py     # Gemini AI integration
│       ├── lightrag_service.py   # Strategic analysis
│       └── supabase_service.py   # Database operations
├── test_n8n_notifications.py     # Port 8001 server
├── requirements.txt
└── .env                          # Configuration
```

### Adding New Endpoints

1. Add route to `app/main.py`
2. Use existing services (`supabase_service`, `lightrag_service`, `gemini_service`)
3. Test with both development servers running

### Debugging Workflows

1. Check N8N execution logs
2. Monitor notification server output (port 8001)
3. Verify database state between steps
4. Use `/api/workflows/health` for system status