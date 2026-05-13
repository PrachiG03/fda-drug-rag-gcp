from google.cloud import storage
import os

PROJECT_ID = "healthcare-rag-prachi"  # your project ID
BUCKET_NAME = "healthcare-rag-raw-data"

def create_bucket_and_upload():
    client = storage.Client(project=PROJECT_ID)
    
    # Create bucket (only once)
    try:
        bucket = client.create_bucket(BUCKET_NAME, location="US")
        print(f"Created bucket: {BUCKET_NAME}")
    except Exception:
        bucket = client.bucket(BUCKET_NAME)
        print(f"Bucket exists, using: {BUCKET_NAME}")

        # Upload CSV to GCS raw zone
    blob = bucket.blob("raw/drug_labels_full.csv")
    blob.upload_from_filename("data/drug_labels_full.csv")
    print(f"Uploaded to: gs://{BUCKET_NAME}/raw/drug_labels_full.csv")

if __name__ == "__main__":
    create_bucket_and_upload()