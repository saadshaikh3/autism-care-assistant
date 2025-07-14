from tavily import AsyncTavilyClient
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import TextNode, NodeWithScore
from config import TAVILY_API_KEY

class WebSearchTool:

    def __init__(self):
        self.client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

    async def web_search(self, query: str) -> str:
        result = await self.client.search( 
            query=query,
            search_depth="advanced",
            include_images=True,
            include_image_descriptions=True,
            include_domains=["nhs.uk"],
            time_range="year"
        )

        # Create source nodes compatible with RAG tool output
        source_nodes = []
        for i, search_result in enumerate(result.get("results", [])):
            # Create a TextNode with web search result data
            text_node = TextNode(
                text=search_result.get("content", ""),
                metadata={
                    "file_name": search_result.get("title", f"Web Result {i+1}"),
                    "url": search_result.get("url", ""),
                    "title": search_result.get("title", ""),
                    "source_type": "web_search"
                }
            )
            
            # Create NodeWithScore to match RAG tool output format
            node_with_score = NodeWithScore(
                node=text_node,
                score=float(search_result.get("score", 0.0))
            )
            source_nodes.append(node_with_score)

        # Create response object that matches expected format
        class WebSearchResponse:
            def __init__(self, query, results, images, source_nodes):
                self.query = query
                self.results = results
                self.images = images
                self.source_nodes = source_nodes
                
            def __str__(self):
                # Return a formatted summary for the LLM
                summary = f"Web search results for '{self.query}':\n\n"
                for i, result in enumerate(self.results[:5]):  # Show top 5 results
                    summary += f"{i+1}. {result.get('title', 'No Title')}\n"
                    summary += f"   URL: {result.get('url', '')}\n"
                    summary += f"   Content: {result.get('content', '')}\n\n"
                
                if self.images:
                    summary += f"\nFound {len(self.images)} related images from NHS sources:\n"
                    for i, img_data in enumerate(self.images[:5]):  # Show top 5 images
                        if isinstance(img_data, dict):
                            # Image data is an object with url and description
                            img_url = img_data.get('url', '')
                            img_desc = img_data.get('description', '')
                            summary += f"- Image {i+1}: {img_url}\n"
                            if img_desc:
                                summary += f"  Description: {img_desc}\n"
                        else:
                            # Fallback for string URLs (backward compatibility)
                            summary += f"- Image {i+1}: {img_data}\n"
                    summary += "\n"
                
                summary += f"Total results: {len(self.results)} web pages, {len(self.images)} images"
                return summary
        
        response = WebSearchResponse(
            query=query,
            results=result.get("results", []),
            images=result.get("images", []),
            source_nodes=source_nodes
        )
        
        return response

    def as_function_tool(self) -> FunctionTool:

        async def tool_fn(query: str) -> str:
            return await self.web_search(query)
        
        return FunctionTool.from_defaults(
            async_fn=tool_fn,
            name="web_search",
            description="Use for answering questions using up-to-date information from the web."
        )

