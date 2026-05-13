import streamlit as st
import pandas as pd
from rag_pipeline import run_rag_query

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="💊",
    layout="wide"
)

st.title("💊 FDA Drug Intelligence Assistant")
st.caption("Built on GCP | Powered by Vertex AI + Gemini | RAG Architecture")

tab1, tab2, tab3 = st.tabs(["Ask a Question", "Drug Explorer", "Architecture"])

with tab1:
    st.subheader("Ask about any drug, interaction, or side effect")
    
    # Example questions
    st.write("**Try these examples:**")
    examples = [
        "What are the drug interactions for warfarin?",
        "What are warnings for diabetes medications?",
        "Side effects of ACE inhibitors?",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            st.session_state.query = ex
    
    query = st.text_input(
        "Your question:",
        value=st.session_state.get("query", ""),
        placeholder="e.g. What drugs interact with metformin?"
    )
    
    if st.button("Get Answer", type="primary") and query:
        with st.spinner("Searching FDA drug labels..."):
            result = run_rag_query(query)
        
        st.success("Answer")
        st.write(result["answer"])
        
        with st.expander("Sources used"):
            for src in result["sources"]:
                st.write(f"- {src}")
        
        st.info("Answers are grounded in real FDA drug label data only.")

with tab2:
    st.subheader("Explore Drug Database")
    df = pd.read_csv("data/drug_labels.csv")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Drugs", len(df))
    col2.metric("Manufacturers", df["manufacturer"].nunique())
    col3.metric("Data Source", "openFDA API")
    
    search = st.text_input("Search drug name:")
    if search:
        filtered = df[df["brand_name"].str.contains(search, case=False, na=False)]
    else:
        filtered = df
    
    st.dataframe(
        filtered[["brand_name", "generic_name", "manufacturer"]].head(20),
        use_container_width=True
    )
with tab3:
    st.subheader("System Architecture")
    st.markdown("""
    **Data Pipeline:**
    - `openFDA API` → Raw drug label JSON → `Google Cloud Storage` (data lake)
    - `GCS` → ETL (Python/Pandas) → `BigQuery` (data warehouse)
    
    **AI/RAG Pipeline:**  
    - Drug text → `Vertex AI Embeddings` (textembedding-gecko) → `ChromaDB` vector store
    - User query → embedding → semantic search → top 3 chunks → `Gemini 1.5 Flash` → answer
    
    **BI Layer:**
    - BigQuery → `Power BI` dashboard (drug analytics, interaction patterns)
    
    **Stack:** Python, GCP (GCS + BigQuery + Vertex AI + Gemini), ChromaDB, Streamlit
    """)