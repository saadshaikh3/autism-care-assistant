from pydantic import BaseModel
from typing import List, Self
from llama_index.core.llms import ChatMessage, MessageRole

class Message(BaseModel):
    sender: str  # 'user' or 'assistant'
    content: str
    
    def to_llamaindex_chatmessage(self) -> ChatMessage:
        """Convert to LlamaIndex ChatMessage format."""
        role = MessageRole.USER if self.sender == "user" else MessageRole.ASSISTANT
        return ChatMessage(role=role, content=self.content)
    
    @classmethod
    def from_llamaindex_chatmessage(cls, chat_message: ChatMessage) -> Self:
        """Create Message from LlamaIndex ChatMessage."""
        sender = "user" if chat_message.role == MessageRole.USER else "assistant"
        # Extract text content from blocks or use content directly
        content = ""
        if chat_message.content:
            content = str(chat_message.content)
        
        return cls(sender=sender, content=content)

class ChatSession(BaseModel):
    assistant: str  # 'uk' or 'india'
    messages: List[Message]
    use_web_search: bool = False
    use_rag: bool = False

class ToolConfig(BaseModel):
    use_web_search: bool = False
    use_rag: bool = False

# Utility functions for chat history conversion
def messages_to_llamaindex_chat_history(messages: List[Message]) -> List[ChatMessage]:
    """Convert list of Message objects to LlamaIndex ChatMessage list."""
    return [msg.to_llamaindex_chatmessage() for msg in messages]

def llamaindex_chat_history_to_messages(chat_messages: List[ChatMessage]) -> List[Message]:
    """Convert LlamaIndex ChatMessage list to Message objects."""
    return [Message.from_llamaindex_chatmessage(chat_msg) for chat_msg in chat_messages]
