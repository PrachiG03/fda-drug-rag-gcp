\# FDA Drug Intelligence RAG System

\*\*Live Demo:\*\* \[huggingface.co/spaces/AIAnalystPrachi/fda-drug-assistant](https://huggingface.co/spaces/AIAnalystPrachi/fda-drug-assistant)



\## What this does

AI-powered healthcare assistant that answers clinical questions 

grounded in real FDA drug label data. Built end-to-end on GCP.



\## Architecture

openFDA API → GCS (data lake) → BigQuery (warehouse) → 

Vertex AI Embeddings → ChromaDB → Gemini 2.5 Flash → Streamlit



\## Stack

Python | GCP (BigQuery, Vertex AI, Cloud Functions, GCS) | 

Gemini 2.5 Flash | ChromaDB | LangChain | Streamlit | 

Power BI (DAX, DirectQuery) | XGBoost | SHAP | Make.com



\## ML Results

\- XGBoost black box warning classifier: AUC \[paste your number]

\- SHAP explainability: warning\_length identified as top predictor

\- TF-IDF + K-Means: 8 drug therapeutic clusters identified



\## Data

\- 40,000+ FDA drug labels across 10 therapeutic categories

\- Source: openFDA API (open.fda.gov)

\- Pipeline: automated ingestion → BigQuery → Power BI DirectQuery

