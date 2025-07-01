#!/usr/bin/env python3
"""
N8N Workflow Communication Server
Handles workflow transitions and triggers strategic analysis
Run with: python test_n8n_notifications.py
"""

import os
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import json
import requests
import asyncio
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="N8N Workflow Communication Server")

# Configuration from environment variables
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://calvinsendawula.app.n8n.cloud")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
NOTIFICATION_SERVER_URL = os.getenv("NOTIFICATION_SERVER_URL", "http://localhost:8001")

# Workflow webhook paths
WF07_WEBHOOK_PATH = "/webhook/generate-summaries"
WF08_WEBHOOK_PATH = "/webhook/filter-content"
WF09_WEBHOOK_PATH = "/webhook/strategic-analysis"
WF12_WEBHOOK_PATH = "/webhook/trigger-pipeline"

async def trigger_workflow_09(article_ids: list, session_id: str = None):
    """Trigger WF-09 Strategic Analysis via webhook"""
    webhook_url = f"{N8N_BASE_URL}{WF09_WEBHOOK_PATH}"
    
    trigger_payload = {
        "article_ids": article_ids,
        "priority": "normal",
        "stage": "analysis_ready",
        "trigger_source": "WF-08_completion",
        "session_id": session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        logger.info(f"🚀 Triggering WF-09 at: {webhook_url}")
        logger.info(f"📋 Payload: {trigger_payload}")
        
        response = requests.post(
            webhook_url,
            json=trigger_payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info("✅ WF-09 triggered successfully!")
            return {"status": "success", "response": response.json()}
        else:
            logger.error(f"❌ WF-09 trigger failed: {response.status_code} - {response.text}")
            return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
            
    except Exception as e:
        logger.error(f"❌ Error triggering WF-09: {e}")
        return {"status": "error", "message": str(e)}


async def trigger_workflow_07(article_ids: list, session_id: str = None):
    """Trigger WF-07 Summary Generation via webhook"""
    webhook_url = f"{N8N_BASE_URL}{WF07_WEBHOOK_PATH}"
    
    trigger_payload = {
        "article_ids": article_ids,
        "session_id": session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "priority": "normal",
        "trigger_source": "inter_workflow",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        logger.info(f"🚀 Triggering WF-07 at: {webhook_url}")
        response = requests.post(webhook_url, json=trigger_payload, timeout=60)
        
        if response.status_code == 200:
            logger.info("✅ WF-07 triggered successfully!")
            return {"status": "success", "response": response.json()}
        else:
            logger.error(f"❌ WF-07 trigger failed: {response.status_code} - {response.text}")
            return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        logger.error(f"❌ Error triggering WF-07: {e}")
        return {"status": "error", "message": str(e)}


async def trigger_workflow_08(article_ids: list, session_id: str = None):
    """Trigger WF-08 Content Filtering via webhook"""
    webhook_url = f"{N8N_BASE_URL}{WF08_WEBHOOK_PATH}"
    
    trigger_payload = {
        "article_ids": article_ids,
        "session_id": session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "workflow": "WF-08",
        "summaries_generated": len(article_ids),
        "next_stage": "content_filtering",
        "triggered_by": "inter_workflow",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        logger.info(f"🚀 Triggering WF-08 at: {webhook_url}")
        response = requests.post(webhook_url, json=trigger_payload, timeout=60)
        
        if response.status_code == 200:
            logger.info("✅ WF-08 triggered successfully!")
            return {"status": "success", "response": response.json()}
        else:
            logger.error(f"❌ WF-08 trigger failed: {response.status_code} - {response.text}")
            return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        logger.error(f"❌ Error triggering WF-08: {e}")
        return {"status": "error", "message": str(e)}


async def trigger_workflow_12(article_ids: list, session_id: str = None):
    """Trigger WF-12 Pipeline Orchestration via webhook"""
    webhook_url = f"{N8N_BASE_URL}{WF12_WEBHOOK_PATH}"
    
    trigger_payload = {
        "article_ids": article_ids,
        "session_id": session_id or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "priority": "normal",
        "trigger_source": "inter_workflow",
        "manual_trigger": True,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        logger.info(f"🚀 Triggering WF-12 at: {webhook_url}")
        response = requests.post(webhook_url, json=trigger_payload, timeout=30)
        
        if response.status_code == 200:
            logger.info("✅ WF-12 triggered successfully!")
            return {"status": "success", "response": response.json()}
        else:
            logger.error(f"❌ WF-12 trigger failed: {response.status_code} - {response.text}")
            return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        logger.error(f"❌ Error triggering WF-12: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/n8n/webhook/sources-discovered")
async def sources_discovered(request: Request):
    """Receive source discovery notifications from N8N WF-01"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🎉 [{timestamp}] SOURCE DISCOVERY NOTIFICATION:")
        print(f"   Workflow: {payload.get('workflow', 'Unknown')}")
        print(f"   Flow: {payload.get('flow', 'Unknown')}")
        print(f"   Sources Found: {payload.get('sources_found', 0)}")
        print(f"   Search Context: {payload.get('search_context', 'Unknown')}")
        print(f"   Summary: {payload.get('summary', 'No summary')}")
        print(f"   Timestamp: {payload.get('timestamp', 'Unknown')}")
        print("-" * 50)
        
        return {
            "status": "success", 
            "message": "Source discovery notification received",
            "received_at": timestamp
        }
        
    except Exception as e:
        print(f"❌ Error processing notification: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/n8n/webhook/categories-updated")
async def categories_updated(request: Request):
    """Receive category update notifications from N8N WF-03"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n📚 [{timestamp}] CATEGORIES UPDATE NOTIFICATION:")
        print(f"   Workflow: {payload.get('workflow', 'Unknown')}")
        print(f"   Categories Updated: {payload.get('categories_updated', 0)}")
        print(f"   Research Period: {payload.get('research_period', 'Unknown')}")
        print(f"   Summary: {payload.get('summary', 'No summary')}")
        print(f"   Timestamp: {payload.get('timestamp', 'Unknown')}")
        print("-" * 50)
        
        return {
            "status": "success", 
            "message": "Categories update notification received",
            "received_at": timestamp
        }
        
    except Exception as e:
        print(f"❌ Error processing notification: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/n8n/webhook/summaries-complete")
async def summaries_complete(request: Request):
    """Simulate receiving webhook response from WF-07 completion"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n📝 [{timestamp}] WF-07 SUMMARY GENERATION COMPLETED:")
        print(f"   Status: {payload.get('status', 'unknown')}")
        print(f"   Summaries Generated: {payload.get('summaries_generated', 0)}")
        print(f"   Articles Processed: {payload.get('articles_processed', 0)}")
        print(f"   Next Stage: {payload.get('next_stage', 'unknown')}")
        print("-" * 60)
        
        print("🚀 THIS IS WHERE WF-08 CONTENT FILTERING WOULD BE TRIGGERED:")
        print("   Simulating trigger with the following data:")
        trigger_data = {
            "article_ids": payload.get("article_ids", []),
            "stage": "filtering_ready",
            "trigger_source": "WF-07_completion"
        }
        print(f"   Trigger Data: {trigger_data}")
        print(f"   Target Webhook: http://your-n8n-domain/webhook/filter-content")
        print("-" * 60)
        
        return {
            "status": "success", 
            "message": "WF-07 completion acknowledged, WF-08 would be triggered here",
            "received_at": timestamp,
            "simulation": "WF-08_trigger_simulated"
        }
        
    except Exception as e:
        print(f"❌ Error processing notification: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/n8n/webhook/filtering-complete")
async def filtering_complete(request: Request):
    """Receive WF-08 completion and trigger WF-09 Strategic Analysis"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"\n🔍 [{timestamp}] WF-08 CONTENT FILTERING COMPLETED:")
        logger.info(f"   Status: {payload.get('status', 'unknown')}")
        logger.info(f"   Total Processed: {payload.get('total_processed', 0)}")
        logger.info(f"   Approved Count: {payload.get('approved_count', 0)}")
        logger.info(f"   Next Stage: {payload.get('next_stage', 'unknown')}")
        logger.info("-" * 60)
        
        # Extract approved article IDs
        approved_ids = payload.get("approved_ids", [])
        session_id = payload.get("session_id", None)
        
        if not approved_ids:
            logger.warning("⚠️  No approved articles to process")
            return {
                "status": "warning", 
                "message": "No approved articles for analysis",
                "received_at": timestamp
            }
        
        # Actually trigger WF-09
        logger.info("🚀 TRIGGERING WF-09 STRATEGIC ANALYSIS:")
        trigger_result = await trigger_workflow_09(approved_ids, session_id)
        
        if trigger_result["status"] == "success":
            logger.info("✅ WF-09 triggered successfully!")
            return {
                "status": "success", 
                "message": "WF-08 completion acknowledged, WF-09 triggered",
                "received_at": timestamp,
                "trigger_result": trigger_result,
                "articles_queued": len(approved_ids)
            }
        else:
            logger.error("❌ Failed to trigger WF-09")
            return {
                "status": "error", 
                "message": "WF-08 acknowledged but WF-09 trigger failed",
                "received_at": timestamp,
                "error": trigger_result
            }
        
    except Exception as e:
        logger.error(f"❌ Error processing notification: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/n8n/webhook/analysis-complete")
async def analysis_complete(request: Request):
    """Receive final completion notification from WF-09"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🎯 [{timestamp}] WF-09 STRATEGIC ANALYSIS COMPLETED:")
        print(f"   Status: {payload.get('status', 'unknown')}")
        print(f"   Session ID: {payload.get('session_id', 'Unknown')}")
        print(f"   Articles Processed: {payload.get('articles_processed', 0)}")
        print(f"   Final Stage: {payload.get('final_stage', 'Unknown')}")
        print("-" * 60)
        
        print("🏁 PIPELINE COMPLETE!")
        print("   All workflows finished successfully!")
        print("   WF-07 → WF-08 → WF-09 ✅")
        print("   No further actions needed.")
        print("-" * 60)
        
        return {
            "status": "success", 
            "message": "Pipeline completed successfully",
            "received_at": timestamp,
            "pipeline_status": "complete"
        }
        
    except Exception as e:
        print(f"❌ Error processing notification: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/n8n/webhook/pipeline-complete")
async def pipeline_complete(request: Request):
    """Receive pipeline completion notifications from WF-12"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🎉 [{timestamp}] WF-12 PIPELINE COMPLETED SUCCESSFULLY:")
        print(f"   Session ID: {payload.get('session_id', 'Unknown')}")
        print(f"   Pipeline Type: {payload.get('pipeline_type', 'Unknown')}")
        print(f"   Duration: {payload.get('duration_minutes', 0)} minutes")
        print(f"   Articles Processed: {payload.get('articles_processed', 0)}")
        print(f"   Performance: {payload.get('performance_rating', 'Unknown')}")
        print("-" * 60)
        
        return {
            "status": "success",
            "message": "Pipeline completion acknowledged",
            "received_at": timestamp
        }
        
    except Exception as e:
        print(f"❌ Error processing completion: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/n8n/webhook/pipeline-error")
async def pipeline_error(request: Request):
    """Receive pipeline error notifications from WF-12"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🚨 [{timestamp}] WF-12 PIPELINE ERROR:")
        print(f"   Session ID: {payload.get('session_id', 'Unknown')}")
        print(f"   Error Stage: {payload.get('error_stage', 'Unknown')}")
        print(f"   Error Message: {payload.get('error_message', 'Unknown')}")
        print("-" * 60)
        
        return {
            "status": "success",
            "message": "Pipeline error acknowledged",
            "received_at": timestamp
        }
        
    except Exception as e:
        print(f"❌ Error processing error notification: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/n8n/webhook/pipeline-warning")
async def pipeline_warning(request: Request):
    """Receive pipeline warning notifications from WF-12"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n⚠️ [{timestamp}] WF-12 PIPELINE WARNING:")
        print(f"   Session ID: {payload.get('session_id', 'Unknown')}")
        print(f"   Warning Stage: {payload.get('warning_stage', 'Unknown')}")
        print(f"   Warning Message: {payload.get('warning_message', 'Unknown')}")
        print("-" * 60)
        
        return {
            "status": "success",
            "message": "Pipeline warning acknowledged",
            "received_at": timestamp
        }
        
    except Exception as e:
        print(f"❌ Error processing warning: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/n8n/webhook/error-log")
async def error_log(request: Request):
    """Receive error notifications from N8N workflows"""
    try:
        payload = await request.json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🚨 [{timestamp}] ERROR NOTIFICATION:")
        print(f"   Workflow: {payload.get('workflow_name', 'Unknown')}")
        print(f"   Error: {payload.get('error_message', 'Unknown error')}")
        print(f"   Execution ID: {payload.get('execution_id', 'Unknown')}")
        print(f"   Timestamp: {payload.get('timestamp', 'Unknown')}")
        print("-" * 50)
        
        return {
            "status": "success", 
            "message": "Error log received",
            "received_at": timestamp
        }
        
    except Exception as e:
        print(f"❌ Error processing error log: {e}")
        return {"status": "error", "message": str(e)}

# === WORKFLOW TRIGGER ENDPOINTS ===

@app.post("/api/trigger/workflow-07")
async def trigger_wf07_endpoint(request: Request):
    """Endpoint to trigger WF-07 Summary Generation"""
    try:
        payload = await request.json()
        article_ids = payload.get("article_ids", [])
        session_id = payload.get("session_id")
        
        if not article_ids:
            raise HTTPException(status_code=400, detail="article_ids are required")
        
        result = await trigger_workflow_07(article_ids, session_id)
        return result
    except Exception as e:
        logger.error(f"Error in WF-07 trigger endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trigger/workflow-08")
async def trigger_wf08_endpoint(request: Request):
    """Endpoint to trigger WF-08 Content Filtering"""
    try:
        payload = await request.json()
        article_ids = payload.get("article_ids", [])
        session_id = payload.get("session_id")
        
        if not article_ids:
            raise HTTPException(status_code=400, detail="article_ids are required")
        
        result = await trigger_workflow_08(article_ids, session_id)
        return result
    except Exception as e:
        logger.error(f"Error in WF-08 trigger endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trigger/workflow-09")
async def trigger_wf09_endpoint(request: Request):
    """Endpoint to trigger WF-09 Strategic Analysis"""
    try:
        payload = await request.json()
        article_ids = payload.get("article_ids", [])
        session_id = payload.get("session_id")
        
        if not article_ids:
            raise HTTPException(status_code=400, detail="article_ids are required")
        
        result = await trigger_workflow_09(article_ids, session_id)
        return result
    except Exception as e:
        logger.error(f"Error in WF-09 trigger endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trigger/workflow-12")
async def trigger_wf12_endpoint(request: Request):
    """Endpoint to trigger WF-12 Pipeline Orchestration"""
    try:
        payload = await request.json()
        article_ids = payload.get("article_ids", [])
        session_id = payload.get("session_id")
        
        if not article_ids:
            raise HTTPException(status_code=400, detail="article_ids are required")
        
        result = await trigger_workflow_12(article_ids, session_id)
        return result
    except Exception as e:
        logger.error(f"Error in WF-12 trigger endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "N8N Workflow Communication Server is running!",
        "description": "Receives workflow completions and triggers strategic analysis",
        "pipeline": "WF-08 → Notifications Server → WF-09 → LightRAG Backend",
        "backend_base_url": BACKEND_BASE_URL,
        "n8n_instance": N8N_BASE_URL,
        "webhook_endpoints": [
            "/api/n8n/webhook/sources-discovered",
            "/api/n8n/webhook/categories-updated", 
            "/api/n8n/webhook/summaries-complete",
            "/api/n8n/webhook/filtering-complete",
            "/api/n8n/webhook/analysis-complete",
            "/api/n8n/webhook/pipeline-complete",
            "/api/n8n/webhook/pipeline-error",
            "/api/n8n/webhook/pipeline-warning",
            "/api/n8n/webhook/error-log"
        ],
        "trigger_endpoints": [
            "/api/trigger/workflow-07",
            "/api/trigger/workflow-08",
            "/api/trigger/workflow-09",
            "/api/trigger/workflow-12"
        ],
        "configuration": {
            "n8n_base_url": N8N_BASE_URL,
            "backend_base_url": BACKEND_BASE_URL,
            "notification_server_url": NOTIFICATION_SERVER_URL
        }
    }

if __name__ == "__main__":
    print("🚀 Starting N8N Workflow Communication Server on http://localhost:8001")
    print("📡 Listening for workflow completion webhooks...")
    print("💡 Will trigger WF-09 Strategic Analysis when WF-08 completes")
    print("🔄 Flow: WF-08 → This Server → WF-09 → LightRAG (port 8000)")
    print(f"🎯 N8N Instance: {N8N_BASE_URL}")
    print(f"🧠 LightRAG Backend: {BACKEND_BASE_URL}")
    print("-" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001) 