from google.cloud import bigquery
import pandas as pd

PROJECT_ID = "healthcare-rag-prachi"
DATASET_ID = "healthcare_rag"
TABLE_ID = "drug_labels"

def load_to_bigquery():
    client = bigquery.Client(project=PROJECT_ID)
    
    # Create dataset
    dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
    dataset.location = "US"
    try:
        client.create_dataset(dataset)
        print(f"Created dataset: {DATASET_ID}")
    except Exception:
        print(f"Dataset exists: {DATASET_ID}")
    
    # Load CSV to BigQuery
    df = pd.read_csv("data/drug_labels_full.csv")
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"Loaded {len(df)} rows to BigQuery: {table_ref}")
    print("✓ Tonight's work done! Data is in GCS + BigQuery.")

if __name__ == "__main__":
    load_to_bigquery()