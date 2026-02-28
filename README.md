# ⚡ Fast Vectorless RAG: Local PageIndex + Groq LPUs

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Groq](https://img.shields.io/badge/Groq-LPU_Inference-orange.svg)
![Architecture](https://img.shields.io/badge/Architecture-Vectorless_RAG-success.svg)

An enterprise-grade, hybrid Retrieval-Augmented Generation (RAG) pipeline that completely bypasses traditional vector databases. 

This project uses the open-source **PageIndex** framework to build a semantic document tree locally (ensuring 100% data privacy during ingestion), and routes the sequential tree-traversal reasoning to **Groq's LPUs** (Llama-3.1-8B) for sub-second inference latency.



## 🧠 The Architecture Problem
Traditional RAG chunks documents into arbitrary text blocks, embeds them, and stores them in a vector database. This destroys the semantic layout of the document (headers, sub-headers, tables) and confuses the LLM on complex queries.

"Vectorless RAG" solves this by reading the document like a human and mapping it into a semantic tree. However, navigating this tree requires sequential agentic reasoning. If you use standard OpenAI GPT-4, this traversal takes 10+ seconds and costs significant API credits. 

## 💡 The Hybrid Solution
1. **Local Privacy:** We execute the PageIndex framework natively on local hardware to ingest the PDF and build the semantic tree. Zero sensitive documents are sent to the cloud.
2. **Hyper-Speed Inference:** We hijack the framework's OpenAI base URL to route the reasoning tasks through Groq's LPUs, reducing inference latency from 10 seconds to **< 1.5 seconds**.



