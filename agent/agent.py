from llama_index.core.agent.workflow import FunctionAgent
from agent.rag_tool import RAGTool
from llama_index.core import Settings
from agent.web_search_tool import WebSearchTool
from config import SYSTEM_PROMPTS, DOCS_DIRECTORY

def build_agent(
    assistant: str = "uk",
    use_rag: bool = False,
    use_web_search: bool = False,
) -> FunctionAgent:
    _tools = []
    # Add RAG tool if enabled
    if use_rag:
        rag_tool = RAGTool(DOCS_DIRECTORY)
        _tools.append(rag_tool.as_query_engine_tool())
    # Add web search tool if enabled
    if use_web_search:
        web_search_tool = WebSearchTool()
        _tools.append(web_search_tool.as_function_tool())

    # Use system prompt from config
    _system_prompt = SYSTEM_PROMPTS.get(assistant, "")

    agent = FunctionAgent(
        tools=_tools,
        llm=Settings.llm,
        system_prompt=_system_prompt,
        verbose=True,
        timeout=300,
        streaming=True
        )
    return agent
