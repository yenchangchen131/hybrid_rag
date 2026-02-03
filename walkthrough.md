Hybrid RAG System Walkthrough
This document outlines the setup, usage, and architecture of the Hybrid RAG (Retrieval-Augmented Generation) system built for the DRCD dataset.

1. System Overview
The system implements a Hybrid Retrieval strategy combining:

Vector Search (Semantic): Using OpenAI Embeddings (text-embedding-3-small) + Cosine Similarity.
Keyword Search (Lexical): Using MongoDB Text Search (BM25 algorithm).
RRF Fusion: Merging results using Reciprocal Rank Fusion.
The backend is MongoDB (storing text and embeddings), and the frontend is Streamlit.

2. Setup & Installation
Ensure you have uv installed.

Start MongoDB:

docker-compose up -d
(Or ensure a local MongoDB is running on port 27017)

Install Dependencies:

uv sync
Environment Variables: Ensure 
.env
 contains:

OPENAI_API_KEY=sk-...
3. Data Ingestion (One-time)
If you need to re-ingest data:

Download & Preprocess:
uv run src/data_preprocess.py
Ingest to MongoDB:
uv run src/ingest_data.py
Generate Embeddings:
uv run src/update_embeddings.py
4. Troubleshooting & Restarting
How to restart after 
drop_db.py
?
If you have cleared the database, follow these steps to restore the system:

Ingest Data (Restore text to MongoDB):
uv run src/ingest_data.py
Generate Embeddings (Restore vectors):
uv run src/update_embeddings.py
Launch App:
uv run streamlit run src/streamlit_app.py
5. Usage
Interactive CLI
uv run main.py
Web Interface (Streamlit)
uv run streamlit run src/streamlit_app.py
Open browser at http://localhost:8501.
To Stop: Press Ctrl+C in the terminal.
5. Evaluation Results
We evaluated the system on 50 validation questions.

Metric	Score	Note
Recall@5	98.0%	49/50 correct docs retrieved in top 5.
Recall@1	70.0%	Correct doc is top 1 in 70% of cases.
MRR@5	0.812	High ranking efficiency.

Comment
Ctrl+Alt+M
