import streamlit as st
import asyncio
import logging
from typing import AsyncGenerator
from agent.agent import build_agent
from models import Message, messages_to_llamaindex_chat_history
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
from config import SYSTEM_PROMPTS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure there is a running event loop
try:
    loop = asyncio.get_running_loop()
    logger.info("Using existing event loop")
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info("Created new event loop")

# Streamlit page configuration
st.set_page_config(
    page_title="Autism Care Assistant 1",
    page_icon="🤝",
    layout="centered",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "assistant": "uk",
        "use_web_search": False,
        "use_rag": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    logger.info(f"Session state initialized: {list(st.session_state.keys())}")

def validate_session_state():
    """Validate that all required session state keys exist"""
    required_keys = ["messages", "assistant", "use_web_search", "use_rag"]
    for key in required_keys:
        if key not in st.session_state:
            logger.error(f"Missing session state key: {key}")
            initialize_session_state()
            return False
    return True

def render_citations(sources, message_index=0):
    """Render source nodes as clickable citation widgets"""
    if not sources:
        return
    
    st.markdown("---")
    st.markdown("**📚 Sources:**")
    
    # Create clickable citation badges with minimal spacing
    cols = st.columns([1] * len(sources))
    for i, source in enumerate(sources):
        citation_num = i + 1
        with cols[i]:
            # Make keys unique by including message index
            button_key = f"citation_badge_{message_index}_{citation_num}"
            state_key = f"show_citation_{message_index}_{citation_num}"
            
            if st.button(f"📄 {citation_num}", key=button_key, help=f"Click to view Source {citation_num}", use_container_width=True):
                # Toggle the citation display
                current_state = st.session_state.get(state_key, False)
                st.session_state[state_key] = not current_state
                st.rerun()
    
    st.markdown("")  # Add some spacing
    
    # Show details for any opened citations
    for i, source in enumerate(sources):
        citation_num = i + 1
        
        # Extract metadata with support for both file-based and web-based sources
        metadata = source.node.metadata
        source_type = metadata.get('source_type', 'file')
        
        if source_type == 'web_search':
            # Web search source - use URL and title
            display_name = metadata.get('title', 'Unknown Web Page')
            source_identifier = metadata.get('url', 'No URL')
            source_label = "🌐 URL"
        else:
            # File-based source - use filename
            display_name = metadata.get('file_name', 'Unknown Source')
            source_identifier = display_name
            source_label = "📁 Filename"
        
        score = getattr(source, 'score', 0.0) if hasattr(source, 'score') else 0.0
        score_percentage = round(score * 100, 1) if score else 0.0
        
        # Create unique keys for this message and citation
        state_key = f"show_citation_{message_index}_{citation_num}"
        text_key = f"text_content_{message_index}_{citation_num}"
        close_key = f"close_citation_{message_index}_{citation_num}"
        
        # Show details if citation was clicked
        if st.session_state.get(state_key, False):
            with st.expander(f"📖 Source {citation_num} - {display_name}", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if source_type == 'web_search':
                        st.markdown(f"**{source_label}:** [{source_identifier}]({source_identifier})")
                    else:
                        st.markdown(f"**{source_label}:** `{source_identifier}`")
                
                with col2:
                    st.markdown(f"**🎯 Relevance:** {score_percentage}%")
                
                st.markdown("**📝 Full Text Content:**")
                # Display full text in a text area with proper width and label
                st.text_area(
                    label="Source Content",
                    value=source.node.text, 
                    height=300,
                    key=text_key,
                    label_visibility="collapsed"
                )
                
                # Close button
                if st.button("❌ Close", key=close_key):
                    st.session_state[state_key] = False
                    st.rerun()


# Initialize session state FIRST
initialize_session_state()

# Validate session state before proceeding
if not validate_session_state():
    st.error("Failed to initialize session state. Please refresh the page.")
    st.stop()

async def get_agent_response(user_input: str, assistant: str, use_rag: bool, use_web_search: bool, messages: list) -> AsyncGenerator[str, None]:
    """Get streaming response from the FunctionAgent"""
    handler = None
    
    try:
        logger.info(f"Processing user input: {user_input[:50]}...")
        
        # Build agent with current settings
        agent = build_agent(
            assistant=assistant,
            use_rag=use_rag,
            use_web_search=use_web_search,
        )
        
        # Convert chat history to the format expected by the agent
        chat_messages = [
            Message(sender=msg["role"], content=msg["content"])
            for msg in messages
            if msg["role"] in ("user", "assistant")
        ]
        chat_history = messages_to_llamaindex_chat_history(chat_messages)
        
        # Create agent context and run
        agent_ctx = Context(agent)
        handler = agent.run(user_input, chat_history=chat_history, ctx=agent_ctx)
        
        # Stream the response with robust error handling
        try:
            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    if event.delta:  # Only yield non-empty deltas
                        yield event.delta
                elif isinstance(event, ToolCallResult):
                    tool_name = event.tool_name
                    tool_input = event.tool_kwargs
                    logger.info(f"Tool executed: {tool_name} with input args: {tool_input}")
                    
                    # Extract source nodes from tool output and store directly in session state
                    if hasattr(event, 'tool_output') and hasattr(event.tool_output, 'raw_output'):
                        raw_output = event.tool_output.raw_output
                        if hasattr(raw_output, 'source_nodes'):
                            # Store sources in a temporary key that we'll move to the correct message index later
                            current_sources = getattr(st.session_state, '_temp_sources', [])
                            current_sources.extend(raw_output.source_nodes)
                            st.session_state._temp_sources = current_sources
                            logger.info(f"Collected {len(raw_output.source_nodes)} source nodes from {tool_name}")
        finally:
            # Clean up the workflow properly after streaming is complete
            if handler:
                await handler.cancel_run()
                logger.info("The stream has been stopped!")
                # Give the workflow a moment to clean up its internal tasks
                await asyncio.sleep(0.2)
        
        logger.info("Response generation completed")
        
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        yield "I apologize, but I encountered an error while processing your request. Please try again."

# Page title and description
st.title("🤝 Autism Care Assistant")
st.markdown("*Supporting parents, teachers, carers, therapists, and professionals caring for autistic children.*")

# Sidebar with native Streamlit components
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Assistant selection using native selectbox
    assistant_options = list(SYSTEM_PROMPTS.keys())
    current_index = assistant_options.index(st.session_state.assistant) if st.session_state.assistant in assistant_options else 0
    
    st.session_state.assistant = st.selectbox(
        "Choose Assistant",
        options=assistant_options,
        index=current_index,
        format_func=lambda x: x.upper(),
        help="Select which assistant personality to use"
    )
    
    st.divider()
    
    # Tool settings using native toggles
    st.subheader("🛠️ Advanced Options")
    
    st.session_state.use_web_search = st.toggle(
        "🌐 NHS Based Answers",
        value=st.session_state.use_web_search,
        help="Get answers based on latest NHS information"
    )
    
    st.session_state.use_rag = st.toggle(
        "📚 Research Based Answers", 
        value=st.session_state.use_rag,
        help="Get answers from research papers and academic publications"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear chat history"):
        # Clear messages
        st.session_state.messages = []
        
        # Clear all citation-related session state keys
        keys_to_remove = []
        for key in st.session_state.keys():
            if any(key.startswith(prefix) for prefix in [
                'show_citation_', 'text_content_', 'close_citation_', 
                'citation_badge_', 'sources_', '_temp_sources'
            ]):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        st.rerun()

# Display chat messages using native chat components
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show citations for assistant messages if available
        if message["role"] == "assistant":
            sources_key = f"sources_{i}"
            if sources_key in st.session_state:
                render_citations(st.session_state[sources_key], message_index=i)

# Handle user input using native chat input
if user_input := st.chat_input("Ask me anything about autism support..."):
    # Validate session state before processing
    if not validate_session_state():
        st.error("Session state error. Please refresh the page.")
        st.stop()
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            # Create a streaming wrapper that handles the thinking indicator
            async def streaming_response():
                async_gen = get_agent_response(
                    user_input,
                    st.session_state.assistant,
                    st.session_state.use_rag,
                    st.session_state.use_web_search,
                    st.session_state.messages
                )
                
                try:
                    async for chunk in async_gen:
                        if chunk and chunk.strip():
                            yield chunk
                        
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield "I apologize, but I encountered an error while processing your request. Please try again."
            
            # Show spinner while waiting for first response
            with st.spinner("🤔 Thinking..."):
                # Stream the response directly
                full_response = st.write_stream(streaming_response())
            
            # Add response to session state
            if full_response and full_response.strip():
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Move temporary sources to the correct message index
                if hasattr(st.session_state, '_temp_sources') and st.session_state._temp_sources:
                    message_index = len(st.session_state.messages) - 1
                    sources_key = f"sources_{message_index}"
                    st.session_state[sources_key] = st.session_state._temp_sources
                    logger.info(f"Stored {len(st.session_state._temp_sources)} sources for message {message_index}")
                    
                    # Clear temporary sources
                    st.session_state._temp_sources = []
                    
                    # Render citations immediately
                    render_citations(st.session_state[sources_key], message_index=message_index)
                
            else:
                fallback_msg = "I'm here to help! Please ask me anything about autism support."
                st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
            
        except Exception as e:
            error_msg = "❌ I'm experiencing technical difficulties. Please try again."
            logger.error(f"Unexpected error: {e}", exc_info=True)
            st.error(f"Error details: {str(e)}")
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
# AI Disclaimer
st.markdown(
    "<div style='background-color: #fffbf0; border: 0px solid #ffeaa7; border-radius: 0.25rem; padding: 0.3rem; font-size: 0.7em; color: #856404; opacity: 0.7; margin-bottom: 1rem;'>"
    "⚠️ <strong>Disclaimer:</strong> This assistant is powered by AI and provides general information only. "
    "Always consult with qualified healthcare professionals or doctors for specific medical concerns about the child's care."
    "</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.7em;'>"
    "© De Montfort University, Leicester, UK | Saad Shaikh | Professor Daniela Romano | Dr. Shobha Sivaramakrishnan  - 2025"
    "</div>",
    unsafe_allow_html=True
)
