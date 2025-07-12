import os
import time
from typing import Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.readers.docling import DoclingReader
from llama_index.core import Settings


class RateLimitedEmbeddingModel(BaseEmbedding):
    base_model: Any
    delay_seconds: float
    last_request_time: float
    
    def __init__(self, base_model, delay_seconds=15, **kwargs):
        """
        Wrapper for embedding model with rate limiting
        delay_seconds: Wait time between requests (15 seconds = 4 requests per minute, staying under 5/min limit)
        """
        super().__init__(
            embed_batch_size=base_model.embed_batch_size,
            callback_manager=base_model.callback_manager,
            base_model=base_model,
            delay_seconds=delay_seconds,
            last_request_time=0,
            **kwargs
        )
    
    def _get_text_embedding(self, text: str) -> list[float]:
        # Calculate time since last request
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # If we need to wait, sleep
        if time_since_last < self.delay_seconds:
            sleep_time = self.delay_seconds - time_since_last
            print(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        # Make the request
        result = self.base_model._get_text_embedding(text)
        self.last_request_time = time.time()
        return result
    
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # For batch requests, process one by one with delays
        results = []
        for i, text in enumerate(texts):
            print(f"Processing embedding {i+1}/{len(texts)}")
            result = self._get_text_embedding(text)
            results.append(result)
        return results
    
    def _get_query_embedding(self, query: str) -> list[float]:
        # Query embedding is the same as text embedding
        return self._get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> list[float]:
        # For async, just use the sync version
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # For async, just use the sync version
        return self._get_text_embeddings(texts)
    
    async def _aget_query_embedding(self, query: str) -> list[float]:
        # For async, just use the sync version
        return self._get_query_embedding(query)

def get_embed_model(api_key: str):
    return GoogleGenAIEmbedding(
    model_name="gemini-embedding-exp-03-07",
    api_key=api_key,
    embedding_config=EmbedContentConfig(task_type="QUESTION_ANSWERING", output_dimensionality=3072)
)

def get_llm_model(api_key: str):
    return GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

def get_or_create_vector_index(docs_dir: str, index_dir: str) -> VectorStoreIndex:
    # Check if index storage exists
    if os.path.exists(index_dir) and os.listdir(index_dir):
        # Load from disk
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context=storage_context)
    else:
        # Build from docs and save
        reader = DoclingReader()
        # documents = SimpleDirectoryReader(docs_dir).load_data()
        documents = SimpleDirectoryReader(input_dir=docs_dir, file_extractor={".pdf": reader}).load_data()

        for doc in documents:
            doc.excluded_llm_metadata_keys = ["file_path", "file_size", "creation_date", "last_modified_date", "last_accessed_date"]
            doc.excluded_embed_metadata_keys = ["file_path", "file_size", "creation_date", "last_modified_date", "last_accessed_date"]
        
        rate_limited_embed_model = RateLimitedEmbeddingModel(Settings.embed_model, delay_seconds=15)
        
        print(f"Creating vector index with {len(documents)} documents (this will take some time due to rate limiting)...")
        index = VectorStoreIndex.from_documents(documents=documents, embed_model=rate_limited_embed_model)
        index.storage_context.persist(persist_dir=index_dir)
    return index

