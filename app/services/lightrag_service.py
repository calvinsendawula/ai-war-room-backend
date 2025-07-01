import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed

logger = logging.getLogger(__name__)

class LightRAGService:
    """
    Service for handling LightRAG vector store operations and strategic analysis.
    Based on user's existing implementation with optimizations for article processing.
    """
    
    def __init__(self, gemini_complete_func):
        self.gemini_complete_func = gemini_complete_func
        self.logger = logger
        
        # Configuration from environment
        self.working_dir = os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_storage")
        self.embedding_model = os.getenv("LIGHTRAG_EMBEDDING_MODEL", "mxbai-embed-large")
        self.embedding_dimension = int(os.getenv("LIGHTRAG_EMBEDDING_DIMENSION", "1024"))
        self.embedding_max_tokens = int(os.getenv("LIGHTRAG_EMBEDDING_MAX_TOKENS", "8192"))
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.cosine_threshold = float(os.getenv("LIGHTRAG_COSINE_THRESHOLD", "0.2"))
        
        self.rag = None
        
        self.logger.info(">>>>>> LightRAG Service initialized <<<<<<")
        self.logger.info(f"Working directory: {self.working_dir}")
        self.logger.info(f"Embedding model: {self.embedding_model}")
        self.logger.info(f"Embedding dimension: {self.embedding_dimension}")

    def is_data_indexed(self) -> bool:
        """
        Check if data is already indexed in the working directory.
        
        Returns:
            bool: True if data appears to be indexed
        """
        working_path = Path(self.working_dir)
        
        # Check for key LightRAG files
        key_files = [
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json", 
            "graph_chunk_entity_relation.graphml"
        ]
        
        existing_files = []
        for file_name in key_files:
            file_path = working_path / file_name
            if file_path.exists() and file_path.stat().st_size > 100:  # File exists and has content
                existing_files.append(file_name)
        
        if len(existing_files) >= 2:  # At least 2 key files exist
            self.logger.info(f"Found existing index files: {existing_files}")
            return True
        
        self.logger.info("No existing index files found")
        return False

    async def initialize(self) -> None:
        """
        Initialize LightRAG with configuration settings.
        """
        self.logger.info(f"Initializing LightRAG with working directory: {self.working_dir}")
        
        # Ensure working directory exists
        working_path = Path(self.working_dir)
        working_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Working directory ready: {self.working_dir}")

        # Setup embedding function
        def embedding_func(texts, embed_model=self.embedding_model, host=self.ollama_host) -> List[List[float]]:
            return ollama_embed(texts, embed_model=embed_model, host=host)

        # Initialize LightRAG (simplified like LAMA)
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.gemini_complete_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dimension,
                max_token_size=self.embedding_max_tokens,
                func=embedding_func,
            ),
            vector_storage="FaissVectorDBStorage",
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": self.cosine_threshold
            }
        )

        # Initialize storages
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
        self.logger.info("LightRAG initialized successfully")

    async def insert_article(self, article: Dict[str, Any]) -> None:
        """
        Insert a single article into the LightRAG vector store.
        
        Args:
            article: Article dictionary with id, title, content, summary, importance
        """
        if not self.rag:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        
        try:
            self.logger.info(f"DEBUG: Starting article insertion for {article['id']}")
            
            # Prepare article content for LightRAG
            article_text = f"""
Title: {article['title']}

Content: {article['content']}

Summary: {article['summary']}

Strategic Importance: {article['importance']}

Article ID: {article['id']}
"""
            
            self.logger.info(f"DEBUG: About to call rag.ainsert() for {article['id']}")
            self.logger.info(f"DEBUG: Article text length: {len(article_text)}")
            
            # Insert into LightRAG
            await self.rag.ainsert(article_text)
            self.logger.info(f"Inserted article into LightRAG: {article['id']}")
            
        except Exception as e:
            self.logger.error(f"Error inserting article {article['id']}: {e}")
            self.logger.error(f"Full traceback:", exc_info=True)
            raise

    async def insert_articles_batch(self, articles: List[Dict[str, Any]]) -> None:
        """
        Insert multiple articles with concurrency optimization.
        
        Args:
            articles: List of article dictionaries
        """
        if not self.rag:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        
        # Process articles with controlled concurrency
        semaphore = asyncio.Semaphore(3)  # Process 3 articles concurrently
        
        async def insert_single(article):
            async with semaphore:
                await self.insert_article(article)
        
        # Execute batch insertion
        tasks = [insert_single(article) for article in articles]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info(f"Batch insertion completed for {len(articles)} articles")

    async def query_strategic_insights(self, query_text: str, article_id: str = None) -> Dict[str, Any]:
        """
        Query LightRAG for strategic insights and analysis.
        
        Args:
            query_text: The strategic analysis query
            article_id: Optional article ID for context
            
        Returns:
            Dict containing strategic analysis results
        """
        if not self.rag:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        
        try:
            # Use hybrid mode for comprehensive analysis
            response = await self.rag.aquery(query_text, param=QueryParam(mode="hybrid"))
            
            # Structure the response for strategic analysis
            analysis = {
                "strategic_takeaway": self._extract_strategic_takeaway(response),
                "impact_analysis": self._extract_impact_analysis(response),
                "timing_analysis": self._extract_timing_analysis(response), 
                "players_analysis": self._extract_players_analysis(response),
                "precedent_analysis": self._extract_precedent_analysis(response),
                "full_response": response,
                "query_mode": "hybrid",
                "article_id": article_id
            }
            
            self.logger.info(f"Strategic analysis completed for query: {query_text[:50]}...")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in strategic analysis: {e}")
            return {
                "error": str(e),
                "strategic_takeaway": "Analysis failed",
                "impact_analysis": "",
                "timing_analysis": "",
                "players_analysis": "",
                "precedent_analysis": "",
                "full_response": "",
                "article_id": article_id
            }

    async def discover_connections(self, article_id: str) -> List[Dict[str, Any]]:
        """
        Discover connections between articles using LightRAG's relationship capabilities.
        
        Args:
            article_id: The article ID to find connections for
            
        Returns:
            List of connection dictionaries
        """
        if not self.rag:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        
        try:
            # Query for connections and relationships
            connection_query = f"""
            Find all related articles, entities, and strategic connections for article ID: {article_id}
            
            Focus on:
            1. Similar strategic themes
            2. Related companies/organizations
            3. Connected technologies or policies
            4. Timeline relationships
            5. Competitive dynamics
            
            IMPORTANT: Respond in plain text only. Do not use markdown formatting, hashtags (#), bold text (**), or any special formatting. Use simple sentences and paragraphs.
            """
            
            # Use global mode for relationship discovery
            response = await self.rag.aquery(connection_query, param=QueryParam(mode="global"))
            
            # Extract structured connections
            connections = self._parse_connections_from_response(response, article_id)
            
            self.logger.info(f"Found {len(connections)} connections for article {article_id}")
            return connections
            
        except Exception as e:
            self.logger.error(f"Error discovering connections for {article_id}: {e}")
            return []

    def _extract_strategic_takeaway(self, response: str) -> str:
        """Extract strategic takeaway from LightRAG response"""
        # Simple extraction - could be enhanced with more sophisticated parsing
        lines = response.split('\n')
        for line in lines:
            if 'strategic' in line.lower() and ('implication' in line.lower() or 'takeaway' in line.lower()):
                return line.strip()
        return response[:200] + "..." if len(response) > 200 else response

    def _extract_impact_analysis(self, response: str) -> str:
        """Extract impact analysis from response"""
        # Look for impact-related content
        lines = response.split('\n')
        impact_lines = [line for line in lines if 'impact' in line.lower()]
        return ' '.join(impact_lines).strip() if impact_lines else ""

    def _extract_timing_analysis(self, response: str) -> str:
        """Extract timing analysis from response"""
        lines = response.split('\n')
        timing_lines = [line for line in lines if any(word in line.lower() for word in ['timing', 'time', 'when', 'timeline'])]
        return ' '.join(timing_lines).strip() if timing_lines else ""

    def _extract_players_analysis(self, response: str) -> str:
        """Extract players/stakeholders analysis from response"""
        lines = response.split('\n')
        player_lines = [line for line in lines if any(word in line.lower() for word in ['player', 'company', 'organization', 'stakeholder'])]
        return ' '.join(player_lines).strip() if player_lines else ""

    def _extract_precedent_analysis(self, response: str) -> str:
        """Extract precedent analysis from response"""
        lines = response.split('\n')
        precedent_lines = [line for line in lines if any(word in line.lower() for word in ['precedent', 'previous', 'history', 'similar'])]
        return ' '.join(precedent_lines).strip() if precedent_lines else ""

    def _parse_connections_from_response(self, response: str, article_id: str) -> List[Dict[str, Any]]:
        """Parse connection information from LightRAG response"""
        connections = []
        
        # Simple parsing - extract mentioned entities and relationships
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['related', 'connection', 'similar', 'linked']):
                connections.append({
                    "type": "strategic_relationship",
                    "description": line.strip(),
                    "source_article_id": article_id,
                    "confidence": 0.8,  # Default confidence
                    "discovered_via": "lightrag"
                })
        
        return connections[:10]  # Limit to top 10 connections

    async def cleanup(self) -> None:
        """
        Cleanup LightRAG resources.
        """
        if self.rag:
            # LightRAG doesn't have explicit cleanup, but we can log completion
            self.logger.info("LightRAG service cleanup completed")

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the LightRAG index.
        
        Returns:
            Dictionary with index statistics
        """
        working_path = Path(self.working_dir)
        
        stats = {
            "working_dir": self.working_dir,
            "initialized": self.rag is not None,
            "files_present": {}
        }
        
        # Check for key files and their sizes
        key_files = [
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "graph_chunk_entity_relation.graphml"
        ]
        
        for file_name in key_files:
            file_path = working_path / file_name
            if file_path.exists():
                stats["files_present"][file_name] = file_path.stat().st_size
            else:
                stats["files_present"][file_name] = 0
        
        return stats 