from langgraph.graph import StateGraph, END
from typing import Dict, List, Any
import json
import sqlite3
from datetime import datetime
from rag_infer import RAGPipeline
from storage import VectorStore
from embedder import Embedder
from retriever import GraphRAGRetriever

class WorkflowManager:
    def __init__(self, db_path: str = "workflows.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self.rag_pipeline = None
        self.retriever = None

    def _create_tables(self):
        """Initialize SQLite table for storing workflows."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                name TEXT,
                workflow_json TEXT,
                status TEXT,
                created_at TIMESTAMP,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def initialize_rag(self, store: VectorStore, embedder: Embedder, llm_provider: str = "mock", llm_model: str = "llama3"):
        """Initialize the RAG pipeline and retriever for use in workflows."""
        self.retriever = GraphRAGRetriever(store, embedder)
        self.rag_pipeline = RAGPipeline(self.retriever, llm_provider, llm_model)

    def add_workflow(self, workflow_id: str, name: str, workflow_json: Dict, metadata: Dict = {}):
        """Store a workflow in the database."""
        try:
            workflow_str = json.dumps(workflow_json)
            metadata_str = json.dumps(metadata)
            self.conn.execute(
                "INSERT OR REPLACE INTO workflows VALUES (?, ?, ?, ?, ?, ?)",
                (workflow_id, name, workflow_str, "active", datetime.now().isoformat(), metadata_str)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error adding workflow {workflow_id}: {str(e)}")
            return False

    def get_workflow(self, workflow_id: str) -> Dict:
        """Retrieve a workflow by ID."""
        row = self.conn.execute(
            "SELECT workflow_id, name, workflow_json, status, created_at, metadata FROM workflows WHERE workflow_id = ?",
            (workflow_id,)
        ).fetchone()
        if row:
            return {
                "workflow_id": row[0],
                "name": row[1],
                "workflow_json": json.loads(row[2]),
                "status": row[3],
                "created_at": row[4],
                "metadata": json.loads(row[5])
            }
        return None

    def list_workflows(self) -> List[Dict]:
        """List all workflows."""
        rows = self.conn.execute("SELECT workflow_id, name, status, created_at, metadata FROM workflows").fetchall()
        return [
            {
                "workflow_id": row[0],
                "name": row[1],
                "status": row[2],
                "created_at": row[3],
                "metadata": json.loads(row[4])
            }
            for row in rows
        ]

    def build_graph(self, workflow_json: Dict) -> StateGraph:
        """Build a LangGraph from a workflow JSON."""
        graph = StateGraph(Dict[str, Any])

        # Define RAG retrieval node
        def rag_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
            if not self.retriever:
                return {"error": "Retriever not initialized"}
            query = state.get("query", "")
            config = state.get("config", {})
            top_k = config.get("top_k", 3)
            document_ids = config.get("document_ids", None)
            results = self.retriever.retrieve(query, top_k=top_k, document_ids=document_ids)
            return {
                "query": query,
                "retrieved_chunks": results,
                "config": config
            }

        # Define RAG answer node
        def rag_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
            if not self.rag_pipeline:
                return {"error": "RAG pipeline not initialized"}
            query = state.get("query", "")
            answer = self.rag_pipeline.answer(query)
            return {
                "query": query,
                "answer": answer,
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }

        # Add nodes from workflow JSON
        nodes = workflow_json.get("nodes", [])
        for node in nodes:
            node_id = node["id"]
            node_type = node.get("type", "placeholder")
            node_config = node.get("config", {})
            if node_type == "rag_retrieval":
                graph.add_node(node_id, lambda state: rag_retrieval_node({**state, "config": node_config}))
            elif node_type == "rag_answer":
                graph.add_node(node_id, rag_answer_node)
            else:
                def placeholder_node(state: Dict[str, Any]) -> Dict[str, Any]:
                    return {"node_id": node_id, "message": f"Node {node_id} executed", "config": node_config}
                graph.add_node(node_id, placeholder_node)

        # Add edges from workflow JSON
        edges = workflow_json.get("edges", [])
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            graph.add_edge(source, target)

        # Set entry and exit points
        entry_node = workflow_json.get("entry_node", nodes[0]["id"] if nodes else None)
        if entry_node:
            graph.set_entry_point(entry_node)
            for node in nodes:
                if not any(edge["source"] == node["id"] for edge in edges):
                    graph.add_edge(node["id"], END)

        return graph.compile()

    def execute_workflow(self, workflow_id: str, input_data: Dict) -> Dict:
        """Execute a workflow with given input data."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return {"error": f"Workflow {workflow_id} not found"}

        try:
            graph = self.build_graph(workflow["workflow_json"])
            result = graph.invoke(input_data)
            return {"workflow_id": workflow_id, "result": result}
        except Exception as e:
            return {"error": f"Error executing workflow {workflow_id}: {str(e)}"}