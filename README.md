# Medibot 🩺

A Retrieval-Augmented Generation (RAG) medical chatbot that answers health questions using trusted medical reference books as its knowledge base — it only answers from those sources and says so when it doesn't know something.

## How it works

1. **Indexing (offline, one-time):** Medical PDFs in `data/` are loaded, split into chunks, embedded, and stored in a local FAISS vector index.
2. **Query (per question):** The user's question is embedded and matched against the FAISS index to retrieve the most relevant chunks, which are passed to an LLM along with the question to generate a grounded answer.

```
PDF sources → chunks → embeddings → FAISS index
                                         ↓
User question → embed → similarity search → top-k chunks → LLM → answer + sources
```

## Features

- **RAG-based Q&A** grounded in real medical reference books (no hallucinated answers)
- **Source citations** — every answer shows which document/page it came from
- **Accounts** — sign up / log in to save chat history, or use it as a guest (nothing saved)
- **Multiple conversations** — start new chats and revisit past ones from the sidebar, like a normal chat app

## Knowledge base

- *The Gale Encyclopedia of Medicine* — general medical reference
- *Where There Is No Doctor* (Hesperian Health Guides) — practical, common-condition guidance

## Tech stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Orchestration | [LangChain](https://www.langchain.com) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) |
| LLM | `Qwen2.5-7B-Instruct` via Hugging Face Inference API |
| PDF parsing | [pypdf](https://pypdf.readthedocs.io) |
| Auth & chat history | PostgreSQL ([Supabase](https://supabase.com)) + `bcrypt` |

## Project structure

```
├── medibot.py                  # Main Streamlit app
├── database.py                 # User accounts + chat history (Postgres)
├── create_memory_for_llm.py    # Builds the FAISS vector index from data/
├── connect_memory_with_llm.py  # Original CLI prototype (reference only)
├── data/                       # Source medical PDFs
├── vectorstore/db_faiss/       # Pre-built FAISS index
└── requirements.txt
```

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/saloniasthana/medibot.git
cd medibot
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Configure environment variables**

Copy `.env.example` to `.env` and fill in your own values:

```
HF_TOKEN="your_huggingface_access_token"
DATABASE_URL="postgresql://user:password@host:5432/dbname"
```

- `HF_TOKEN` — a [Hugging Face access token](https://huggingface.co/settings/tokens) with "Make calls to Inference Providers" permission enabled
- `DATABASE_URL` — a PostgreSQL connection string (e.g. from [Supabase](https://supabase.com))

**3. Build the vector index** (only needed once, or after adding new PDFs to `data/`)

```bash
python create_memory_for_llm.py
```

**4. Run the app**

```bash
streamlit run medibot.py
```

## Adding more medical documents

Drop any additional text-based PDF into `data/`, then re-run `create_memory_for_llm.py` to rebuild the vector index. The app will pick up the expanded knowledge base on next restart.

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io) — set `HF_TOKEN` and `DATABASE_URL` under the app's **Secrets** in the same TOML format as `.env`.

## Disclaimer

Medibot is an informational tool, not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.
