# MedRAG System Design

## 1. Overview
MedRAG is a modular, production-ready Retrieval-Augmented Generation (RAG) platform for medical information, leveraging multi-agent orchestration, semantic search, and scalable cloud deployment. It is designed for hackathon and real-world healthcare scenarios, supporting locality-aware search, document ingestion, and secure, scalable user access.

## 2. Architecture
- **Frontend:** Streamlit web UI for chat and agent interaction.
- **Backend:** Python (LangChain, LangGraph) orchestrates agents, tools, and workflows.
- **Vector Database:** FAISS (self-hosted) for semantic search and RAG.
- **LLM Integration:** Groq Llama-3.3-70b (via API), with support for other LLMs.
- **Cloud Deployment:** Dockerized, deployable to AWS ECS/EC2/S3/RDS/Cognito/CloudWatch/Route53/ACM/Shield/WAF.

## 3. Agent & Tool Design
- **Agents:**
  - Supervisor: Orchestrates workflow, delegates to sub-agents.
  - Clinical Agent: Handles clinical queries and medical reasoning.
  - Documentation Agent: Ingests and summarizes medical documents.
  - Patient Agent: Interfaces with patient-facing queries.
  - Research Agent: Handles literature and PubMed search.
- **Tools:**
  - Medical calculators, PubMed search, RAG tool, locality-aware doctor search, image/vision analysis, voice module, etc.

## 4. Data Flow & RAG Process
1. User submits query via Streamlit UI.
2. Supervisor agent routes query to appropriate sub-agent(s).
3. Agents use tools (e.g., semantic search, calculators, PubMed) as needed.
4. RAG pipeline retrieves relevant documents from FAISS vector DB.
5. LLM generates response, optionally citing sources.
6. Response returned to user via UI.

## 5. Technology Stack
- **Python 3.9+**
- **LangChain, LangGraph** (agent orchestration)
- **Streamlit** (UI)
- **FAISS** (vector DB)
- **sentence-transformers** (embeddings)
- **DuckDuckGo Search API** (web search)
- **AWS ECS/EC2/S3/RDS/Cognito/CloudWatch/Route53/ACM/Shield/WAF** (cloud infra)
- **Docker** (containerization)
- **Other:** pandas, numpy, scikit-learn, transformers, etc.

## 6. Security & Scalability
- **Authentication:** AWS Cognito/Auth0 for user management.
- **Data Security:** HTTPS, AWS Shield/WAF, encrypted storage.
- **Scalability:** Dockerized microservices, ECS/EC2 auto-scaling, managed RDS.
- **Monitoring:** AWS CloudWatch for logs and metrics.

## 7. Key Design Decisions
- Modular agent/tool design for extensibility.
- Vector-based semantic search for accurate RAG.
- Locality-aware search for healthcare relevance.
- Production-ready cloud deployment with security best practices.

---
For more details, see architecture diagrams and code documentation in the repository.
