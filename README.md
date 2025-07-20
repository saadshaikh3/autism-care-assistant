# autism-care-assistant

**App Name:** Autism Care Assistant  
A compassionate and informative chat assistant to support caregivers of autistic children, including parents, teachers, carers, therapists, and professionals. Regionally tuned for the UK and India, with optional web search and document-based (RAG) help.

---

## 🔧 Features

- **Region‑tuned AI assistant**: Choose between UK or India with customized prompts.
- **Web Search**: Optionally enable live web search using Tavily.
- **RAG Document Search**: Load local support documents and query them via vector-based retrieval.
- **Streaming Chat**: Real-time bot responses as you type.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/autism-care-assistant.git
cd autism-care-assistant
```

### 2. Create a venv and activate it. Then install dependencies using

```bash
pip install -r requirements.txt
```

### 3. Set environment variables
Create a .env file in the root folder based on the env-template file and fill in the envs

### 4. Prepare support documents (optional)
Place any .txt, .pdf, or other supported formats in data/docs/.
The vector index will be created automatically.

### 5. Run locally
```bash
streamlit run app.py
```

You’ll see the chat UI where you can select UK/India assistant, toggle tools, and chat.