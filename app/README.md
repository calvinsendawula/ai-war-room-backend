# AI War Room Backend Setup

## 🚀 Quick Start

### 1. Prerequisites
**MUST BE RUNNING FIRST:**
```bash
# Start Ollama service (for embeddings)
ollama serve

# Pull the embedding model
ollama pull mxbai-embed-large
```

### 2. Environment Setup
Ensure your root `.env` file has all required variables (see main README).

### 3. Install Dependencies
```bash
cd app
pip install -r requirements.txt
```

### 4. Run Backend
```bash
cd app
python run.py
```

Backend will start on `http://localhost:8000`

## 🔄 Integration with N8n

### Update WF-09 Backend URL
Change your N8n WF-09 "Backend Analysis Request" node URL to:
```
http://localhost:8000/api/processing/analyze-single
```

### Port Management
- **NEW Backend**: Port 8000 (for actual processing)
- **Old test_n8n_notifications.py**: Shut down OR run on different port

## 📊 Key Endpoints

- `GET /api/health` - Health check
- `POST /api/processing/analyze-single` - Process single article (WF-09 calls this)
- `POST /api/processing/analyze-batch` - Process multiple articles
- `GET /api/processing/status?session_id=xxx` - Check processing status
- `GET /api/stories/connections/{article_id}` - Get article connections

## 🐛 Troubleshooting

### Common Issues:
1. **Ollama not running**: Ensure `ollama serve` is running first
2. **Model not found**: Run `ollama pull mxbai-embed-large`
3. **Supabase connection**: Check SUPABASE_URL and SUPABASE_SERVICE_KEY
4. **Gemini API**: Verify GEMINI_API_KEY is valid

### Logs:
Backend logs show all processing steps and errors for debugging.

## 🧪 Testing

Use your existing 3 article IDs from WF-08 to test WF-09 integration:
```bash
# Test single article processing
curl -X POST http://localhost:8000/api/processing/analyze-single \
  -H "Content-Type: application/json" \
  -d '{
    "article_id": "2ace056e-7a29-4275-8e06-5df93449c11b",
    "title": "Google and Apple: The Deal", 
    "content": "Article content...",
    "summary_text": "Strategic analysis...",
    "importance_level": "HIGH",
    "session_id": "test_session_001"
  }'
``` 