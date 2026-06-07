# GCP Driver Sleepiness CrewAI Agentic System

A cloud-native, agentic driver sleepiness intervention system deployed on **Google Cloud Platform (GCP)**.

This project receives synchronized driver-state features, estimates fatigue risk, retrieves relevant intervention knowledge from a persistent FAISS vector database, uses **Vertex AI Gemini** to generate an in-cabin intervention, validates the result, updates vector memory when needed, logs the full workflow to **BigQuery**, and returns a structured JSON response with agent-level progress.

The system is designed as a **production-style deployed prototype** for demonstrating:

- Cloud-native AI deployment on GCP
- CrewAI-based multi-agent orchestration
- Retrieval-Augmented Generation (RAG)
- Persistent FAISS vector memory
- Vertex AI Gemini integration
- BigQuery observability and logging
- Safety-aware structured output validation

---

## GCP Driver Sleepiness CrewAI Agentic System
```text

## Project Structure
gcp-driver-sleepiness-crewai-agenticsystem/
│
├── app/
│   ├── api_service.py
│   ├── services.py
│   ├── fatigue_logic.py
│   ├── faiss_rag_retriever.py
│   │
│   └── crew_agents/
│       ├── __init__.py
│       ├── tools.py
│       └── crew_workflow.py
│
├── data/
│   ├── intervention_knowledge_base.jsonl
│   └── faiss_vdb/
│       ├── intervention.index
│       └── intervention_docs.json
│
├── schemas/
│   ├── fatigue_features_schema.json
│   └── agent_decision_logs_schema.json
│
├── Dockerfile
├── requirements.txt
└── README.md

## System Architecture

External Client
curl / Python / REST Client
        |
        v
Cloud Run HTTPS Endpoint
        |
        v
FastAPI REST API
        |
        v
POST /predict
        |
        v
CrewAI Multi-Agent Workflow
        |
        +-------------------------------+
        | 1. Fatigue Analysis Agent      |
        | 2. RAG Retrieval Agent         |
        | 3. Intervention Decision Agent |
        | 4. Safety Validation Agent     |
        | 5. Memory Update Agent         |
        | 6. Logging Agent               |
        +-------------------------------+
        |
        v
Structured JSON Response


## **GCP Architecture**
                           +----------------------+
                           |   External Client    |
                           | curl / Python / API  |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           | Google Cloud Run     |
                           | FastAPI Container    |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           | CrewAI Workflow      |
                           | Multi-Agent System   |
                           +----------+-----------+
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
              v                       v                       v
   +--------------------+   +--------------------+   +--------------------+
   | Fatigue Logic      |   | FAISS Vector DB    |   | Vertex AI Gemini   |
   | Python Scoring     |   | RAG Retrieval      |   | LLM Generation     |
   +--------------------+   +--------------------+   +--------------------+
              |                       |                       |
              +-----------------------+-----------------------+
                                      |
                                      v
                           +----------------------+
                           | BigQuery Logging     |
                           | Features + Decisions |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           | JSON API Response    |
                           +----------------------+

