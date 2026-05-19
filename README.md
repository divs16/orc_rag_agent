# ORC RAG Agent

A command-line RAG (Retrieval-Augmented Generation) agent that answers natural language questions about Oracle Recruiting Cloud (ORC) workflows. It combines knowledge from structured test cases and a ERP knowledge base, using Google Gemini for both embeddings and response generation.

## What it does

Instead of manually searching through spreadsheets and documentation to find recruiting process steps, business rules, or configuration guidelines — you just ask a question and get a direct answer with sources cited.

The agent draws from two knowledge sources simultaneously:

- **Test cases** (`ORC_Test_Cases_Detailed.xlsx`) — step-by-step test procedures covering modules like Job Requisition, Candidate Application, Candidate Selection Process, Job Offer, and Hire
- **Transcript knowledge base** (`orc_knowledge_base.txt`) — business rules and configuration guidelines covering recruiting types, hiring team roles, posting rules, approval workflows, compensation, and more

## How it works

```
Your question
     │
     ▼
Gemini embedding model converts question to a vector
     │
     ▼
Cosine similarity search across all 40 pre-embedded knowledge chunks
     │
     ▼
Top 6 most relevant chunks retrieved (from either or both sources)
     │
     ▼
Gemini 2.5 Flash generates a grounded answer with source citations
```

Embeddings are generated once and cached locally. If either source file changes, the cache invalidates and rebuilds automatically.

## Setup

**Prerequisites:** Python 3.11+, a [Gemini API key](https://aistudio.google.com/app/apikey)

**1. Install dependencies**

```bash
pip install google-genai pandas openpyxl numpy
```

**2. Add your API key**

Create a `.env` file in the project folder:

```
GEMINI_API_KEY=your_api_key_here
```

Or export it in your shell:

```bash
export GEMINI_API_KEY=your_api_key_here
```

**3. Place the knowledge files**

Make sure these two files are in the same folder as `rag_agent.py`:

```
orc-rag-agent/
├── rag_agent.py
├── Test_Cases_.xlsx 
├── your_company_knowledge_base.txt
└── .env
```

## Usage

```bash
python3 rag_agent.py
```

On first run, the agent reads both source files, generates embeddings (~5 seconds), and caches them. Every subsequent run loads from cache and starts instantly.

```
Reading ORC_Test_Cases_Detailed(Sheet1).xlsx...
  29 test cases loaded.
Reading orc_knowledge.txt...
  11 knowledge base sections loaded.
Generating embeddings for 40 total chunks...
Embeddings cached.

Oracle Recruiting Cloud RAG Agent
Type your question (or 'quit' to exit)

You: What recruiting type should I use for a contractor in Norway?

Agent: For a contractor in Oman, you should use the "FT Contractor" recruiting type.
       Oman is a country where XYZ does not have operations but does business
       on a conditional basis (Knowledge Base | Recruiting Types). ...

You: quit
```

Type `quit`, `exit`, or press `Ctrl+C` to exit.

### Custom file paths

You can pass different source files as arguments:

```bash
python3 rag_agent.py path/to/test_cases.xlsx path/to/knowledge.txt
```

## Models used

| Purpose | Model |
|---|---|
| Embeddings | `gemini-embedding-001` |
| Answer generation | `gemini-2.5-flash` |

## Notes

- The embeddings cache is saved as `.embeddings_cache.pkl` and is excluded from version control via `.gitignore`
- Your `.env` file is also excluded from version control — never commit your API key
- Answers cite test case IDs (e.g. `ORC-TC-001.1`) and knowledge base topics so you can trace every claim back to its source
