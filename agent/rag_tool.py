from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from agent.utils import get_or_create_vector_index
from config import VECTOR_INDEX_DIR

class RAGTool:
    def __init__(self, docs_dir: str):
        self._index = get_or_create_vector_index(docs_dir, VECTOR_INDEX_DIR)
        self._query_engine = self._index.as_query_engine(vector_store_query_mode=VectorStoreQueryMode.DEFAULT, similarity_top_k=8, response_mode="compact_accumulate")

    def as_query_engine_tool(self) -> QueryEngineTool:
        # Return a QueryEngineTool for direct agent use
        return QueryEngineTool.from_defaults(
            query_engine=self._query_engine,
            name="rag",
            description="Useful for answering questions from the certified indexed documents and publications. ",
        )

