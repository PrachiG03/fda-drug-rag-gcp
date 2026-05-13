# 04_embed_and_store.py
# Builds the ChromaDB vector store from your drug_labels_full.csv
# Uses google-genai SDK (current, not deprecated)

import chromadb
import pandas as pd
from google import genai
from google.genai import types
import time
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "healthcare-rag-prachi")
LOCATION = "us-central1"


def get_embeddings(texts, client):
    """Embed a batch of texts using Vertex AI text-embedding-004."""
    all_embeddings = []
    batch_size = 5  # Vertex AI limit per call

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings = [e.values for e in response.embeddings]
            all_embeddings.extend(embeddings)
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
            time.sleep(0.3)  # be polite to the API
        except Exception as e:
            print(f"  Error on batch {i}: {e}")
            time.sleep(2)
            # retry once
            try:
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                all_embeddings.extend([e.values for e in response.embeddings])
            except Exception as e2:
                print(f"  Retry failed: {e2}. Skipping batch.")
                all_embeddings.extend([[0.0] * 768] * len(batch))

    return all_embeddings


def build_vector_store():
    # ── 1. Init Vertex AI client ─────────────────────────────────────────────
    print("Connecting to Vertex AI...")
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )

    # ── 2. Load your drug data ────────────────────────────────────────────────
    csv_path = "data/drug_labels_full.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run 01_ingest_fda.py first.")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["full_text"]).reset_index(drop=True)

    print(f"Loaded {len(df)} drug records from CSV")

    df["full_text"] = df["full_text"].str[:1500]
    texts = df["full_text"].tolist()

    # ── 3. Set up ChromaDB (deletes old collection first for clean rebuild) ───
    print("Setting up ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if it exists (clean rebuild)
    try:
        chroma_client.delete_collection("drug_labels")
        print("  Deleted old collection")
    except Exception:
        print("  No existing collection to delete")

    collection = chroma_client.create_collection(
        name="drug_labels",
        metadata={"hnsw:space": "cosine"}
    )
    print("  Created new collection")

    # ── 4. Generate embeddings ────────────────────────────────────────────────
    print(f"\nGenerating embeddings for {len(texts)} drugs...")
    print("(This will take a few minutes)\n")
    embeddings = get_embeddings(texts, client)

    # ── 5. Store in ChromaDB ──────────────────────────────────────────────────
    print("\nStoring in ChromaDB...")
    CHROMA_BATCH = 5000  # safely under the 5461 limit
    metadatas = [{
        "brand_name": str(df.iloc[i].get("brand_name", "")),
        "generic_name": str(df.iloc[i].get("generic_name", "")),
        "manufacturer": str(df.iloc[i].get("manufacturer", "")),
        "search_category": str(df.iloc[i].get("search_category", "")),
    } for i in range(len(texts))]

    for start in range(0, len(texts), CHROMA_BATCH):
        end = min(start + CHROMA_BATCH, len(texts))
        collection.add(
            ids=[f"drug_{i}" for i in range(start, end)],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end]
        )
        print(f"  Stored {end}/{len(texts)} in ChromaDB")

    # ── 6. Verify ─────────────────────────────────────────────────────────────
    count = collection.count()
    print(f"\nDone! ChromaDB now has {count} embeddings stored.")

    if count > 0:
        print("SUCCESS - your chroma_db/ folder is ready.")
        print("Next step: copy chroma_db/ to your fda-drug-assistant folder.")
    else:
        print("WARNING - count is 0. Something went wrong. Check errors above.")


if __name__ == "__main__":
    build_vector_store()