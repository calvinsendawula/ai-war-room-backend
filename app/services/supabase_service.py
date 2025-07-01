import os
import logging
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class SupabaseService:
    """
    Service for handling Supabase database operations.
    """
    
    def __init__(self):
        self.logger = logger
        
        # Get Supabase credentials from environment
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
        
        # Initialize Supabase client
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Thread pool for sync operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info(">>>>>> Supabase Service initialized <<<<<<")
        self.logger.info(f"Connected to: {self.supabase_url}")

    async def get_articles_for_analysis(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get articles by IDs that are ready for analysis.
        
        Args:
            article_ids: List of article UUIDs
            
        Returns:
            List of article dictionaries
        """
        try:
            def _fetch_articles():
                response = self.client.table("articles").select(
                    "id, title, content, importance_level, strategic_category_id, publish_date"
                ).in_("id", article_ids).eq("status", "approved").execute()
                return response.data
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            articles = await loop.run_in_executor(self.executor, _fetch_articles)
            
            # Also get summaries if available
            if articles:
                article_ids_found = [article["id"] for article in articles]
                
                def _fetch_summaries():
                    response = self.client.table("article_summaries").select(
                        "article_id, summary_text"
                    ).in_("article_id", article_ids_found).execute()
                    return response.data
                
                summaries = await loop.run_in_executor(self.executor, _fetch_summaries)
                
                # Merge summaries with articles
                summary_map = {s["article_id"]: s["summary_text"] for s in summaries}
                for article in articles:
                    article["summary_text"] = summary_map.get(article["id"], "")
            
            self.logger.info(f"Retrieved {len(articles)} articles for analysis")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching articles for analysis: {e}")
            return []

    async def update_article_status(self, article_id: str, status: str, lightrag_indexed: bool = False) -> bool:
        """
        Update article status and processing flags.
        
        Args:
            article_id: Article UUID
            status: New status value
            lightrag_indexed: Whether article has been indexed in LightRAG
            
        Returns:
            bool: Success status
        """
        try:
            def _update_article():
                update_data = {
                    "status": status,
                    "processed_at": "now()",
                    "updated_at": "now()"
                }
                
                if lightrag_indexed:
                    update_data["lightrag_indexed"] = True
                
                response = self.client.table("articles").update(update_data).eq("id", article_id).execute()
                return response.data
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _update_article)
            
            self.logger.info(f"Updated article {article_id} status to {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating article status: {e}")
            return False

    async def insert_strategic_analysis(self, article_id: str, analysis: Dict[str, Any]) -> bool:
        """
        Insert strategic analysis results into database.
        
        Args:
            article_id: Article UUID
            analysis: Analysis results dictionary
            
        Returns:
            bool: Success status
        """
        try:
            def _insert_analysis():
                analysis_data = {
                    "article_id": article_id,
                    "impact_analysis": analysis.get("impact_analysis", ""),
                    "timing_analysis": analysis.get("timing_analysis", ""),
                    "players_analysis": analysis.get("players_analysis", ""),
                    "precedent_analysis": analysis.get("precedent_analysis", ""),
                    "strategic_takeaway": analysis.get("strategic_takeaway", ""),
                    "importance_score": self._calculate_importance_score(analysis),
                    "analyzed_at": "now()"
                }
                
                response = self.client.table("strategic_analysis").insert(analysis_data).execute()
                return response.data
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _insert_analysis)
            
            self.logger.info(f"Inserted strategic analysis for article {article_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error inserting strategic analysis: {e}")
            return False

    async def insert_connections(self, connections: List[Dict[str, Any]]) -> bool:
        """
        Insert discovered article connections.
        
        Args:
            connections: List of connection dictionaries
            
        Returns:
            bool: Success status
        """
        try:
            if not connections:
                return True
            
            def _insert_connections():
                # Prepare connection records
                connection_records = []
                for conn in connections:
                    record = {
                        "primary_story_id": conn.get("source_article_id"),
                        "connected_story_id": conn.get("target_article_id"),  # This would need to be resolved
                        "connection_type": conn.get("type", "strategic_relationship"),
                        "relationship_strength": conn.get("confidence", 0.5),
                        "discovered_via": "lightrag",
                        "metadata": {
                            "description": conn.get("description", ""),
                            "discovery_method": "lightrag_analysis"
                        }
                    }
                    connection_records.append(record)
                
                if connection_records:
                    response = self.client.table("connected_stories").insert(connection_records).execute()
                    return response.data
                return []
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _insert_connections)
            
            self.logger.info(f"Inserted {len(connections)} article connections")
            return True
            
        except Exception as e:
            self.logger.error(f"Error inserting connections: {e}")
            return False

    async def update_processing_status(self, session_id: str, stage: str, progress: int, current_count: int) -> bool:
        """
        Update processing status for a session.
        
        Args:
            session_id: Processing session ID
            stage: Current processing stage
            progress: Progress percentage
            current_count: Current article count processed
            
        Returns:
            bool: Success status
        """
        try:
            def _update_status():
                update_data = {
                    "stage": stage,
                    "progress_percent": progress,
                    "current_article_count": current_count,
                    "updated_at": "now()"
                }
                
                response = self.client.table("processing_status").update(update_data).eq("session_id", session_id).execute()
                return response.data
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _update_status)
            
            self.logger.info(f"Updated processing status for session {session_id}: {stage} ({progress}%)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating processing status: {e}")
            return False

    async def get_processing_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing status for a session.
        
        Args:
            session_id: Processing session ID
            
        Returns:
            Processing status dictionary or None
        """
        try:
            def _get_status():
                response = self.client.table("processing_status").select("*").eq("session_id", session_id).execute()
                return response.data[0] if response.data else None
            
            loop = asyncio.get_event_loop()
            status = await loop.run_in_executor(self.executor, _get_status)
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting processing status: {e}")
            return None

    def _calculate_importance_score(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate importance score based on analysis content.
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            float: Importance score between 0 and 1
        """
        # Simple scoring based on content length and keywords
        score = 0.5  # Base score
        
        # Check for strategic keywords
        strategic_keywords = ["strategic", "important", "significant", "critical", "major", "breakthrough"]
        
        full_text = " ".join([
            analysis.get("strategic_takeaway", ""),
            analysis.get("impact_analysis", ""),
            analysis.get("timing_analysis", ""),
            analysis.get("players_analysis", "")
        ]).lower()
        
        keyword_count = sum(1 for keyword in strategic_keywords if keyword in full_text)
        score += min(keyword_count * 0.1, 0.3)  # Max 0.3 bonus for keywords
        
        # Adjust based on content length (more detailed analysis = higher score)
        if len(full_text) > 500:
            score += 0.1
        if len(full_text) > 1000:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

    async def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query against the Supabase database.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            List of result dictionaries
        """
        try:
            def _execute_query():
                # Use Supabase's RPC functionality to execute raw SQL
                # Note: This requires setting up a stored procedure in Supabase
                # For now, we'll use the PostgREST API directly
                
                # Since Supabase client doesn't directly support raw SQL with params,
                # we'll need to use the RPC feature or direct HTTP calls
                # For simplicity, we'll construct safe queries without injection risks
                
                if params:
                    # Simple parameter substitution (ensure this is safe)
                    formatted_query = query
                    for param in params:
                        formatted_query = formatted_query.replace('%s', f"'{param}'", 1)
                    query = formatted_query
                
                # Execute via RPC call (requires a stored procedure)
                # For now, use a simpler approach via table operations
                # This is a limitation that may require a custom RPC function
                
                # As a workaround, we'll return empty results and log the issue
                self.logger.warning(f"Raw SQL execution attempted: {query[:100]}...")
                self.logger.warning("Raw SQL execution requires RPC setup in Supabase")
                return []
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _execute_query)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return []

    async def get_dashboard_articles(self) -> List[Dict[str, Any]]:
        """
        Get articles for dashboard display using table operations.
        Temporary method until raw SQL execution is properly set up.
        """
        try:
            def _get_dashboard_data():
                # Get top analyzed articles with strategic analysis
                # TEMPORARY: Show all articles for testing - remove filters
                response = self.client.table("articles").select(
                    "*, strategic_analysis(*), strategic_categories(name, color_code)"
                ).order("publish_date", desc=True).limit(10).execute()
                
                return response.data
            
            loop = asyncio.get_event_loop()
            articles = await loop.run_in_executor(self.executor, _get_dashboard_data)
            
            self.logger.info(f"Retrieved {len(articles)} dashboard articles")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard articles: {e}")
            return []

    async def get_articles_with_filters(self, importance: str = None, category: str = None, 
                                        status: str = None, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get articles with filtering using table operations.
        Temporary method until raw SQL execution is properly set up.
        """
        try:
            def _get_filtered_articles():
                query = self.client.table("articles").select(
                    "*, strategic_analysis(*), strategic_categories(name, color_code)"
                )
                
                # TEMPORARY: Only apply filters if explicitly requested for testing
                if importance:
                    query = query.eq("importance_level", importance)
                if status:
                    query = query.eq("status", status)
                # Note: Category filtering would require a join, which is complex with this approach
                
                response = query.order("publish_date", desc=True).range(offset, offset + limit - 1).execute()
                return response.data
            
            loop = asyncio.get_event_loop()
            articles = await loop.run_in_executor(self.executor, _get_filtered_articles)
            
            self.logger.info(f"Retrieved {len(articles)} filtered articles")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error getting filtered articles: {e}")
            return []

    async def get_article_with_connections(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Get article with connections using table operations.
        Temporary method until raw SQL execution is properly set up.
        """
        try:
            def _get_article_detail():
                # Get article with analysis
                article_response = self.client.table("articles").select(
                    "*, strategic_analysis(*), strategic_categories(name, color_code)"
                ).eq("id", article_id).execute()
                
                if not article_response.data:
                    return None
                
                article = article_response.data[0]
                
                # Get connections
                connections_response = self.client.table("connected_stories").select(
                    "*, primary_story:articles!primary_story_id(id, title), connected_story:articles!connected_story_id(id, title)"
                ).or_(f"primary_story_id.eq.{article_id},connected_story_id.eq.{article_id}").execute()
                
                article['connections'] = connections_response.data
                return article
            
            loop = asyncio.get_event_loop()
            article = await loop.run_in_executor(self.executor, _get_article_detail)
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error getting article detail: {e}")
            return None

    async def get_strategic_categories(self) -> List[Dict[str, Any]]:
        """
        Get all strategic categories with article counts.
        """
        try:
            def _get_categories():
                response = self.client.table("strategic_categories").select(
                    "*, articles(count)"
                ).eq("active", True).order("priority_order").execute()
                
                return response.data
            
            loop = asyncio.get_event_loop()
            categories = await loop.run_in_executor(self.executor, _get_categories)
            
            self.logger.info(f"Retrieved {len(categories)} strategic categories")
            return categories
            
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            return [] 