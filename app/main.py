#!/usr/bin/env python3
"""
AI War Room Backend - FastAPI application with LightRAG integration
"""

import asyncio
import logging
import os
import time
import re
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import requests
from datetime import datetime
from fastapi import Request

# Load environment variables from root directory
root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

from services.lightrag_service import LightRAGService
from services.supabase_service import SupabaseService
from services.gemini_service import GeminiService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://calvinsendawula.app.n8n.cloud")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
NOTIFICATION_SERVER_URL = os.getenv("NOTIFICATION_SERVER_URL", "http://localhost:8001")

# Global services
lightrag_service: Optional[LightRAGService] = None
supabase_service: Optional[SupabaseService] = None
gemini_service: Optional[GeminiService] = None

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)


def clean_article_content(content: str) -> str:
    """
    Clean article content by removing arXiv prefixes and other unwanted formatting.
    
    Pattern: arXiv:YYYY.NNNNNvN Announce Type: replace Abstract: [content]
    """
    if not content:
        return content
    
    # Remove arXiv prefix pattern
    # Pattern: arXiv:YYYY.NNNNNvN Announce Type: replace Abstract: 
    arxiv_pattern = r'^arXiv:\d{4}\.\d{5}v\d+\s+Announce Type:\s*replace\s+Abstract:\s*'
    cleaned_content = re.sub(arxiv_pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove any other common prefixes that might appear
    # You can extend this as you find more patterns
    other_patterns = [
        r'^Abstract:\s*',  # Standalone "Abstract:" 
        r'^Summary:\s*',   # Standalone "Summary:"
    ]
    
    for pattern in other_patterns:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up any extra whitespace
    cleaned_content = cleaned_content.strip()
    
    return cleaned_content


def clean_article_data(article: Dict[str, Any]) -> Dict[str, Any]:
    """Clean all content fields in an article dictionary before sending to frontend"""
    if article and 'content' in article:
        article['content'] = clean_article_content(article['content'])
    return article


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global lightrag_service, supabase_service, gemini_service
    
    logger.info("=== AI War Room Backend Starting ===")
    
    # Initialize services
    try:
        # Initialize Gemini service
        gemini_service = GeminiService()
        
        # Initialize Supabase service
        supabase_service = SupabaseService()
        
        # Create bulletproof wrapper function for LightRAG
        def create_llm_wrapper():
            async def llm_wrapper(*args, **kwargs):
                try:
                    logger.info(f"DEBUG: LLM wrapper called with args={len(args)}, kwargs={list(kwargs.keys()) if kwargs else 'None'}")
                    
                    # Handle different calling patterns
                    if args:
                        prompt = args[0] if args else ""
                        # Remove prompt from kwargs if it exists
                        kwargs.pop('prompt', None)
                    else:
                        prompt = kwargs.pop('prompt', "")
                    
                    logger.info(f"DEBUG: Extracted prompt: '{prompt[:50]}...'")
                    
                    # Call the actual Gemini service
                    result = await gemini_service.gemini_complete(prompt, **kwargs)
                    logger.info(f"DEBUG: LLM wrapper returning result length: {len(result)}")
                    return result
                    
                except Exception as e:
                    logger.error(f"DEBUG: LLM wrapper error: {e}")
                    logger.error(f"Full traceback:", exc_info=True)
                    return f"LLM Error: {str(e)}"
            
            return llm_wrapper
        
        # Initialize LightRAG service with bulletproof wrapper
        lightrag_service = LightRAGService(create_llm_wrapper())
        await lightrag_service.initialize()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("=== AI War Room Backend Shutting Down ===")
    if lightrag_service:
        await lightrag_service.cleanup()


# Pydantic models
class ArticleAnalysisRequest(BaseModel):
    article_id: str
    title: str
    content: str
    summary_text: str
    importance_level: str
    session_id: str
    priority: str = "normal"


class BatchAnalysisRequest(BaseModel):
    article_ids: List[str]
    priority: str = "normal"
    session_id: str
    callback_webhook: Optional[str] = None


class AnalysisResponse(BaseModel):
    status: str
    article_id: str
    connections: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    processing_time: float


class BatchAnalysisResponse(BaseModel):
    status: str
    session_id: str
    processed_count: int
    articles_queued: int
    estimated_completion: str


# Frontend Data Models
class DashboardData(BaseModel):
    top_stories: List[Dict[str, Any]]
    strategic_threads: List[Dict[str, Any]]
    recent_analysis: List[Dict[str, Any]]
    processing_status: Dict[str, Any]


# Create FastAPI app
app = FastAPI(
    title="AI War Room Backend",
    description="Strategic Intelligence Analysis with LightRAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "lightrag": lightrag_service is not None,
            "supabase": supabase_service is not None,
            "gemini": gemini_service is not None
        },
        "timestamp": time.time()
    }


# ============= FRONTEND DATA ENDPOINTS =============

@app.get("/api/dashboard")
async def get_dashboard_data():
    """Get complete dashboard data for the frontend"""
    if not supabase_service:
        raise HTTPException(status_code=500, detail="Supabase service not initialized")
    
    try:
        # Get top stories using Supabase service
        top_stories = await supabase_service.get_dashboard_articles()
        
        # Clean content for all stories
        top_stories = [clean_article_data(story) for story in top_stories]
        
        # Get strategic threads (simplified for now)
        strategic_threads = []  # Will be populated when connections exist
        
        # Get recent analysis activity (simplified)
        recent_analysis = top_stories[:5]  # Use top 5 as recent analysis for now
        
        # Get current processing status
        processing_status = {
            "active_sessions": 0,
            "articles_in_pipeline": 0,
            "last_update": time.time()
        }
        
        return DashboardData(
            top_stories=top_stories,
            strategic_threads=strategic_threads,
            recent_analysis=recent_analysis,
            processing_status=processing_status
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data retrieval failed: {str(e)}")


@app.get("/api/articles")
async def get_articles(
    importance: Optional[str] = Query(None, description="Filter by importance level"),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, description="Number of articles to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get articles with filtering and pagination for frontend"""
    if not supabase_service:
        raise HTTPException(status_code=500, detail="Supabase service not initialized")
    
    try:
        # Get articles using Supabase service
        articles = await supabase_service.get_articles_with_filters(
            importance=importance,
            category=category,
            status=status,
            limit=limit,
            offset=offset
        )
        
        # Clean content for all articles
        articles = [clean_article_data(article) for article in articles]
        
        return {"articles": articles, "total": len(articles)}
        
    except Exception as e:
        logger.error(f"Error getting articles: {e}")
        raise HTTPException(status_code=500, detail=f"Articles retrieval failed: {str(e)}")


@app.get("/api/articles/{article_id}")
async def get_article_detail(article_id: str):
    """Get detailed article information for frontend"""
    if not supabase_service:
        raise HTTPException(status_code=500, detail="Supabase service not initialized")
    
    try:
        # Get article with connections using Supabase service
        article = await supabase_service.get_article_with_connections(article_id)
        
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        # Clean content
        article = clean_article_data(article)
        
        return article
        
    except Exception as e:
        logger.error(f"Error getting article detail: {e}")
        raise HTTPException(status_code=500, detail=f"Article detail retrieval failed: {str(e)}")


@app.get("/api/categories")
async def get_categories():
    """Get all strategic categories for frontend"""
    if not supabase_service:
        raise HTTPException(status_code=500, detail="Supabase service not initialized")
    
    try:
        categories = await supabase_service.get_strategic_categories()
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=f"Categories retrieval failed: {str(e)}")


# ============= PROCESSING ENDPOINTS =============

@app.post("/api/processing/analyze-single", response_model=AnalysisResponse)
async def analyze_single_article(request: ArticleAnalysisRequest):
    """
    Analyze a single article with LightRAG and return strategic connections
    """
    start_time = time.time()
    
    if not lightrag_service:
        raise HTTPException(status_code=500, detail="LightRAG service not initialized")
    
    try:
        logger.info(f"Processing article: {request.article_id}")
        
        # Clean content before processing
        cleaned_content = clean_article_content(request.content)
        
        # Insert article into LightRAG for relationship discovery
        await lightrag_service.insert_article({
            "id": request.article_id,
            "title": request.title,
            "content": cleaned_content,
            "summary": request.summary_text,
            "importance": request.importance_level
        })
        
        # Query for strategic connections and insights
        strategic_query = f"""
        Based on this article "{request.title}", provide a strategic analysis in PLAIN TEXT format (no markdown, no hashtags, no formatting):
        
        1. Strategic implications: What strategic implications does this have?
        2. Entity connections: What connections exist to other stories or entities?
        3. Key players: What are the key players and their relationships?
        4. Timing factors: What timing/precedent factors are important?
        
        Article summary: {request.summary_text}
        
        IMPORTANT: Respond in plain text only. Do not use markdown formatting, hashtags (#), bold text (**), or any special formatting. Use simple sentences and paragraphs.
        """
        
        analysis_result = await lightrag_service.query_strategic_insights(
            strategic_query, 
            article_id=request.article_id
        )
        
        # Discover connections to other articles
        connections = await lightrag_service.discover_connections(request.article_id)
        
        # Update article status in Supabase
        if supabase_service:
            await supabase_service.update_article_status(
                request.article_id, 
                "analyzed", 
                lightrag_indexed=True
            )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Article {request.article_id} processed in {processing_time:.2f}s")
        
        return AnalysisResponse(
            status="completed",
            article_id=request.article_id,
            connections=connections,
            analysis=analysis_result,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing article {request.article_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/processing/analyze-batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Queue batch analysis of multiple articles
    """
    if not supabase_service:
        raise HTTPException(status_code=500, detail="Supabase service not initialized")
    
    try:
        # Get articles from Supabase
        articles = await supabase_service.get_articles_for_analysis(request.article_ids)
        
        if not articles:
            raise HTTPException(status_code=404, detail="No approved articles found")
        
        # Start background processing
        background_tasks.add_task(
            process_batch_background,
            articles,
            request.session_id,
            request.priority
        )
        
        estimated_completion = time.time() + (len(articles) * 30)  # 30 seconds per article
        
        logger.info(f"Queued {len(articles)} articles for batch processing")
        
        return BatchAnalysisResponse(
            status="queued",
            session_id=request.session_id,
            processed_count=0,
            articles_queued=len(articles),
            estimated_completion=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(estimated_completion))
        )
        
    except Exception as e:
        logger.error(f"Error queuing batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch queuing failed: {str(e)}")


async def process_batch_background(articles: List[Dict], session_id: str, priority: str):
    """Background task for processing article batches"""
    logger.info(f"Starting background batch processing for session {session_id}")
    
    processed_count = 0
    
    # Process articles with concurrency control
    semaphore = asyncio.Semaphore(2)  # Process 2 articles concurrently
    
    async def process_single(article):
        nonlocal processed_count
        async with semaphore:
            try:
                # Create analysis request
                request = ArticleAnalysisRequest(
                    article_id=article["id"],
                    title=article["title"],
                    content=article["content"],
                    summary_text=article.get("summary_text", ""),
                    importance_level=article.get("importance_level", "MEDIUM"),
                    session_id=session_id,
                    priority=priority
                )
                
                # Process the article
                await analyze_single_article(request)
                
                processed_count += 1
                logger.info(f"Batch progress: {processed_count}/{len(articles)} articles processed")
                
                # Update processing status
                if supabase_service:
                    progress = int((processed_count / len(articles)) * 100)
                    await supabase_service.update_processing_status(
                        session_id, 
                        "analysis_in_progress", 
                        progress, 
                        processed_count
                    )
                
            except Exception as e:
                logger.error(f"Error processing article {article['id']} in batch: {e}")
    
    # Process all articles
    tasks = [process_single(article) for article in articles]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Mark batch as complete
    if supabase_service:
        await supabase_service.update_processing_status(
            session_id, 
            "analysis_complete", 
            100, 
            processed_count
        )
    
    logger.info(f"Batch processing complete for session {session_id}: {processed_count}/{len(articles)} processed")


@app.get("/api/processing/status")
async def get_processing_status(session_id: str):
    """Get processing status for a session"""
    if not supabase_service:
        raise HTTPException(status_code=500, detail="Supabase service not initialized")
    
    try:
        status = await supabase_service.get_processing_status(session_id)
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@app.get("/api/stories/connections/{article_id}")
async def get_article_connections(article_id: str):
    """Get connections for a specific article"""
    if not lightrag_service:
        raise HTTPException(status_code=500, detail="LightRAG service not initialized")
    
    try:
        connections = await lightrag_service.discover_connections(article_id)
        return {"article_id": article_id, "connections": connections}
        
    except Exception as e:
        logger.error(f"Error getting connections for {article_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Connection discovery failed: {str(e)}")


# === ORCHESTRATION ENDPOINTS ===

@app.post("/api/trigger/manual-pipeline")
async def trigger_manual_pipeline(request: Request):
    """Handle frontend button click to trigger WF-12"""
    try:
        payload = await request.json()
        user_id = payload.get('user_id', 'frontend_user')
        priority = payload.get('priority', 'high')
        
        # Send to WF-12 webhook
        webhook_data = {
            "test_mode": False,
            "priority": priority,
            "trigger_source": "frontend_manual",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Triggering manual pipeline for user {user_id} with priority {priority}")
        
        response = requests.post(
            f"{N8N_BASE_URL}/webhook/trigger-pipeline",
            json=webhook_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success", 
                "message": "Pipeline triggered successfully",
                "session_id": result.get('session_id', 'unknown'),
                "webhook_response": result
            }
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Webhook trigger failed: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error triggering pipeline: {e}")
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error triggering manual pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline trigger failed: {str(e)}")


@app.get("/api/pipeline/status/{session_id}")
async def get_pipeline_status(session_id: str):
    """Check status of running pipeline"""
    if not supabase_service:
        raise HTTPException(status_code=500, detail="Supabase service not initialized")
    
    try:
        # Query pipeline_runs table
        result = await supabase_service.execute_query(
            "SELECT * FROM pipeline_runs WHERE session_id = %s ORDER BY started_at DESC LIMIT 1",
            (session_id,)
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Pipeline session not found")
        
        pipeline = result[0]
        
        # Calculate status
        status = "unknown"
        if pipeline.get('completed_at'):
            status = "completed" if pipeline.get('success') else "failed"
        elif pipeline.get('started_at'):
            status = "running"
        
        return {
            "status": "success", 
            "pipeline": {
                **pipeline,
                "current_status": status,
                "duration_minutes": pipeline.get('duration_minutes'),
                "articles_processed": pipeline.get('articles_processed', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting pipeline status for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@app.get("/api/workflows/health")
async def check_workflow_health():
    """Check if all workflows are healthy"""
    if not supabase_service:
        raise HTTPException(status_code=500, detail="Supabase service not initialized")
    
    try:
        # Check recent pipeline runs
        result = await supabase_service.execute_query(
            "SELECT * FROM pipeline_runs WHERE started_at >= NOW() - INTERVAL '24 hours' ORDER BY started_at DESC LIMIT 10"
        )
        
        # Check recent articles
        articles_result = await supabase_service.execute_query(
            "SELECT COUNT(*) as count, status FROM articles WHERE created_at >= NOW() - INTERVAL '24 hours' GROUP BY status"
        )
        
        # Calculate health metrics
        total_runs = len(result)
        successful_runs = len([r for r in result if r.get('success')])
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        # Test N8N webhook connectivity
        webhook_health = "unknown"
        try:
            test_response = requests.get(
                f"{N8N_BASE_URL}/webhook/trigger-pipeline",
                timeout=5
            )
            webhook_health = "healthy" if test_response.status_code in [200, 405] else "unhealthy"
        except:
            webhook_health = "unreachable"
        
        health_status = {
            "overall_health": "healthy" if success_rate >= 80 and webhook_health == "healthy" else "degraded",
            "pipeline_runs": {
                "total_24h": total_runs,
                "successful_24h": successful_runs,
                "success_rate": round(success_rate, 2)
            },
            "articles_24h": {item['status']: item['count'] for item in articles_result},
            "webhook_connectivity": webhook_health,
            "services": {
                "supabase": "healthy" if supabase_service else "unhealthy",
                "lightrag": "healthy" if lightrag_service else "unhealthy",
                "gemini": "healthy" if gemini_service else "unhealthy"
            },
            "recent_runs": result[:5]  # Last 5 runs
        }
        
        return {"status": "success", "health": health_status}
        
    except Exception as e:
        logger.error(f"Error checking workflow health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# === ADDITIONAL WORKFLOW TRIGGER ENDPOINTS ===

@app.post("/api/workflows/trigger/summary-generation")
async def trigger_summary_generation(request: Request):
    """Trigger WF-07 Summary Generation workflow"""
    try:
        payload = await request.json()
        
        webhook_data = {
            "article_ids": payload.get("article_ids", []),
            "session_id": payload.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "priority": payload.get("priority", "normal"),
            "trigger_source": payload.get("trigger_source", "backend_manual"),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Triggering WF-07 Summary Generation with {len(webhook_data['article_ids'])} articles")
        
        response = requests.post(
            f"{N8N_BASE_URL}/webhook/generate-summaries",
            json=webhook_data,
            timeout=60
        )
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "WF-07 Summary Generation triggered",
                "session_id": webhook_data["session_id"],
                "webhook_response": response.json()
            }
        else:
            raise HTTPException(status_code=response.status_code, detail=f"WF-07 trigger failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error triggering WF-07: {e}")
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error triggering WF-07: {e}")
        raise HTTPException(status_code=500, detail=f"WF-07 trigger failed: {str(e)}")


@app.post("/api/workflows/trigger/content-filtering")
async def trigger_content_filtering(request: Request):
    """Trigger WF-08 Content Filtering workflow"""
    try:
        payload = await request.json()
        
        webhook_data = {
            "article_ids": payload.get("article_ids", []),
            "session_id": payload.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "workflow": payload.get("workflow", "WF-08"),
            "summaries_generated": payload.get("summaries_generated", len(payload.get("article_ids", []))),
            "next_stage": "content_filtering",
            "triggered_by": payload.get("triggered_by", "backend_manual"),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Triggering WF-08 Content Filtering with {len(webhook_data['article_ids'])} articles")
        
        response = requests.post(
            f"{N8N_BASE_URL}/webhook/filter-content",
            json=webhook_data,
            timeout=60
        )
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "WF-08 Content Filtering triggered",
                "session_id": webhook_data["session_id"],
                "webhook_response": response.json()
            }
        else:
            raise HTTPException(status_code=response.status_code, detail=f"WF-08 trigger failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error triggering WF-08: {e}")
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error triggering WF-08: {e}")
        raise HTTPException(status_code=500, detail=f"WF-08 trigger failed: {str(e)}")


@app.post("/api/workflows/trigger/strategic-analysis")
async def trigger_strategic_analysis(request: Request):
    """Trigger WF-09 Strategic Analysis workflow"""
    try:
        payload = await request.json()
        
        webhook_data = {
            "article_ids": payload.get("article_ids", []),
            "priority": payload.get("priority", "normal"),
            "stage": "analysis_ready",
            "trigger_source": payload.get("trigger_source", "backend_manual"),
            "session_id": payload.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Triggering WF-09 Strategic Analysis with {len(webhook_data['article_ids'])} articles")
        
        response = requests.post(
            f"{N8N_BASE_URL}/webhook/strategic-analysis",
            json=webhook_data,
            timeout=120
        )
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "WF-09 Strategic Analysis triggered",
                "session_id": webhook_data["session_id"],
                "webhook_response": response.json()
            }
        else:
            raise HTTPException(status_code=response.status_code, detail=f"WF-09 trigger failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error triggering WF-09: {e}")
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error triggering WF-09: {e}")
        raise HTTPException(status_code=500, detail=f"WF-09 trigger failed: {str(e)}")


@app.post("/api/workflows/trigger/sequence")
async def trigger_workflow_sequence(request: Request):
    """Trigger a sequence of workflows (WF-07 → WF-08 → WF-09) with proper chaining"""
    try:
        payload = await request.json()
        article_ids = payload.get("article_ids", [])
        session_id = payload.get("session_id", f"sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        priority = payload.get("priority", "normal")
        
        if not article_ids:
            raise HTTPException(status_code=400, detail="article_ids are required")
        
        logger.info(f"Starting workflow sequence for session {session_id} with {len(article_ids)} articles")
        
        # This mimics what WF-12 does - it triggers the sequence
        sequence_data = {
            "article_ids": article_ids,
            "session_id": session_id,
            "priority": priority,
            "trigger_source": "backend_sequence",
            "sequence_mode": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Trigger WF-12 orchestrator which will handle the sequence
        response = requests.post(
            f"{N8N_BASE_URL}/webhook/trigger-pipeline",
            json=sequence_data,
            timeout=30
        )
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "Workflow sequence initiated via WF-12",
                "session_id": session_id,
                "article_count": len(article_ids),
                "sequence": "WF-07 → WF-08 → WF-09",
                "webhook_response": response.json()
            }
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Sequence trigger failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error triggering sequence: {e}")
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error triggering workflow sequence: {e}")
        raise HTTPException(status_code=500, detail=f"Sequence trigger failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 