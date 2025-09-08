# Agentic RAG Chatbot (MCP-based)

An **agent-based Retrieval-Augmented Generation (RAG)** chatbot implemented as a local Streamlit app.  
Agents communicate using a lightweight Model Context Protocol (MCP) style message format. The system ingests documents (PDF/TXT), builds a FAISS vectorstore with Sentence-Transformers embeddings, and generates answers locally using Flan-T5.

---

## 🌟 Features

- **Multi-format (demo)**: PDF, TXT (extendable to DOCX / PPTX / CSV)  
- **Agent-based design**:
  - `Coordinator` — orchestrates the flow
  - `IngestionAgent` — extracts & cleans text
  - `RetrievalAgent` — chunking, embeddings, FAISS vectorstore
  - `LLMResponseAgent` — builds prompt and runs Flan-T5
- **RAG pipeline**: semantic chunking (overlap), FAISS retrieval, local LLM generation
- **Streamlit UI**: file upload, chat interface, session-based multi-turn conversation

---

## 🧾 Quick Start (local)

> Tested with Python 3.10+ — adjust if you use a different interpreter.


`

2. Create and activate a virtual environment
```bash
python -m venv .venv
# mac / linux
source .venv/bin/activate
# windows (powershell)
.venv\Scripts\Activate.ps1
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run streamlit app
```bash
streamlit run simple_app.py
# or if your repo uses a different entrypoint:
# streamlit run main.py
```

Open `http://localhost:8501` in your browser.

---

## 🛠️ Requirements (example)

Add these into your `requirements.txt` (versions used in development):

```
streamlit==1.22.0
PyPDF2==3.0.1
langchain==0.x
langchain-community==0.x
sentence-transformers==2.2.2
transformers==4.31.0
torch==2.1.0
faiss-cpu==1.7.4
```

> Note: Replace `langchain` and `langchain-community` versions with the ones you tested. If you used `langchain_community` imports, ensure that package is installed (pip package name may vary; check your environment). If you plan to use a Hugging Face model with GPU, install the appropriate CUDA build of `torch`.

---

## 🚀 How it works (short)

1. **User uploads** a document through Streamlit.  
2. UI sends a MCP-style `DOC_UPLOAD` message to the `Coordinator`.  
3. `Coordinator` forwards to `IngestionAgent` → extracts & cleans text.  
4. `IngestionAgent` sends `DOC_INGESTED` to `RetrievalAgent`.  
5. `RetrievalAgent`:  
   - chunks text (500 tokens with overlap)  
   - computes embeddings with `sentence-transformers/all-MiniLM-L6-v2`  
   - builds a FAISS vectorstore  
6. When a question arrives, `Coordinator` sends `QUESTION` to `RetrievalAgent`:  
   - retrieves top-k chunks  
   - sends `CONTEXT_RESPONSE` to `LLMResponseAgent`  
7. `LLMResponseAgent` composes prompt and runs **Flan-T5** (Hugging Face pipeline) and returns `ANSWER` to UI.  
8. UI displays the answer and saves chat history in session state.

---

## 📁 Project layout (reflects your repo)

```
.
├── app/
│   ├── simple_app.py        # Streamlit UI + Coordinator
│   ├── run_app.py / main.py # optional entrypoints
├── data/                    # optional storage
├── docs/                    # architecture slides / images
├── sample_docs/             # example PDF/TXT files
├── tests/                   # unit / integration tests
├── requirements.txt
├── README.md
```

---

## 🔧 Key code/snippet explanation

### MCP message format
Messages created with `create_message(sender, receiver, type, payload)`:
```json
{
  "sender": "RetrievalAgent",
  "receiver": "LLMResponseAgent",
  "type": "CONTEXT_RESPONSE",
  "trace_id": "uuid",
  "payload": {
    "top_chunks": ["..."],
    "query": "..."
  }
}
```

### Important agents (brief)
- **IngestionAgent**: loads document (PDF/TXT), extracts text using `PyPDF2` or plain read  
- **RetrievalAgent**: uses `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=100), `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`, and FAISS wrapper to store texts  
- **LLMResponseAgent**: uses `pipeline("text2text-generation", model="google/flan-t5-base")` and `HuggingFacePipeline` to run the prompt locally  

---

## ⚠️ Known limitations 

- **Context limits**: Flan-T5 has small token window → heavy chunking needed  
- **Retrieval dependency**: accuracy depends on chunking & embeddings  
- **Latency/resources**: Local inference is slow on CPU; GPU recommended  
- **Index persistence**: FAISS currently in-memory; rebuild required on restart  
- **Parser edge cases**: PDF parsing may include headers/footers/noise

- Challenges faced
- The main challenges were managing heavy dependencies (PyTorch, Transformers) and GitHub file-size issues due to the venv folder. Integrating FAISS for retrieval and handling large documents with chunking required careful setup. Running Flan-T5 locally on CPU also caused slow responses, so caching and optimizations were added.

---

## ✅ Suggested next steps

- Persist the vectorstore (ChromaDB or disk-backed FAISS) and store metadata  
- Add a reranker (cross-encoder) for better top-k retrieval quality  
- Add unit and integration tests + a CI workflow (GitHub Actions)
- Deployment with cloud and gpu will create amazing results also 

---



---

## 📝 Usage tips

- If embeddings or FAISS fail to build, check that `sentence-transformers` and `faiss-cpu` are installed.  
- For faster inference, run on GPU with CUDA-enabled PyTorch.  
- Clean noisy text from PDFs by filtering headers/footers.

---

## Acknowledgments

- Hugging Face & Transformers  
- Sentence Transformers  
- FAISS / vector search community  
- Streamlit team  

