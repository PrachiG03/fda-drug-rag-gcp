# 01_ingest_fda.py
# Fetches 4000 VALID records per category = ~40K clean records total
# Skips records with null brand_name AND null generic_name

import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FDA_API_KEY", "")
BASE_URL = "https://api.fda.gov/drug/label.json"

PER_CATEGORY_LIMIT = 4000  # 4000 valid records per category x 10 = ~40K total

def fetch_all_drug_labels():
    all_drugs = []
    limit = 1000  # max per API call with API key

    search_terms = [
        "indications_and_usage:diabetes",
        "indications_and_usage:cancer",
        "indications_and_usage:hypertension",
        "indications_and_usage:pain",
        "indications_and_usage:infection",
        "indications_and_usage:depression",
        "indications_and_usage:cholesterol",
        "indications_and_usage:asthma",
        "indications_and_usage:arthritis",
        "indications_and_usage:heart",
    ]

    for term in search_terms:
        category = term.split("indications_and_usage:")[-1].strip()
        category_count = 0
        skip = 0
        print(f"\nFetching: {category} (target: {PER_CATEGORY_LIMIT} valid records)")

        while category_count < PER_CATEGORY_LIMIT:
            params = {"search": term, "limit": limit, "skip": skip}
            if API_KEY:
                params["api_key"] = API_KEY

            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                if resp.status_code != 200:
                    print(f"  API returned {resp.status_code}, stopping.")
                    break
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    print(f"  No more results for {category}.")
                    break

                for r in results:
                    if category_count >= PER_CATEGORY_LIMIT:
                        break

                    openfda = r.get("openfda", {})
                    brand = openfda.get("brand_name", [""])[0]
                    generic = openfda.get("generic_name", [""])[0]

                    # ── SKIP nulls — only keep records with at least a generic name ──
                    if not brand and not generic:
                        continue

                    drug = {
                        "set_id": r.get("set_id", ""),
                        "id": r.get("id", ""),
                        "brand_name": brand,
                        "generic_name": generic,
                        "manufacturer": openfda.get("manufacturer_name", [""])[0],
                        "product_type": openfda.get("product_type", [""])[0],
                        "route": openfda.get("route", [""])[0],
                        "substance_name": openfda.get("substance_name", [""])[0],
                        "rxcui": str(openfda.get("rxcui", [""])[0]),
                        "application_number": str(openfda.get("application_number", [""])[0]),
                        "indications": " ".join(r.get("indications_and_usage", [""])),
                        "warnings": " ".join(r.get("warnings", r.get("warnings_and_cautions", [""]))),
                        "dosage": " ".join(r.get("dosage_and_administration", [""])),
                        "drug_interactions": " ".join(r.get("drug_interactions", [""])),
                        "adverse_reactions": " ".join(r.get("adverse_reactions", [""])),
                        "contraindications": " ".join(r.get("contraindications", [""])),
                        "pregnancy_category": " ".join(r.get("pregnancy", r.get("teratogenic_effects", [""]))),
                        "pediatric_use": " ".join(r.get("pediatric_use", [""])),
                        "geriatric_use": " ".join(r.get("geriatric_use", [""])),
                        "overdosage": " ".join(r.get("overdosage", [""])),
                        "mechanism_of_action": " ".join(r.get("mechanism_of_action", [""])),
                        "clinical_pharmacology": " ".join(r.get("clinical_pharmacology", [""])),
                        "storage_conditions": " ".join(r.get("storage_and_handling", [""])),
                        "search_category": category,
                    }

                    # ── ML features ──────────────────────────────────────────
                    drug["warning_length"] = len(drug["warnings"])
                    drug["interaction_length"] = len(drug["drug_interactions"])
                    drug["adverse_length"] = len(drug["adverse_reactions"])
                    drug["has_interactions"] = 1 if drug["drug_interactions"] else 0
                    drug["has_black_box"] = 1 if "boxed warning" in drug["warnings"].lower() or "black box" in drug["warnings"].lower() else 0
                    drug["is_prescription"] = 1 if "prescription" in drug["product_type"].lower() else 0
                    drug["has_pediatric_info"] = 1 if drug["pediatric_use"] else 0
                    drug["has_geriatric_info"] = 1 if drug["geriatric_use"] else 0
                    drug["has_pregnancy_info"] = 1 if drug["pregnancy_category"] else 0
                    drug["has_overdosage_info"] = 1 if drug["overdosage"] else 0

                    # ── Full text for RAG ─────────────────────────────────────
                    drug["full_text"] = f"""Drug: {drug['brand_name']} ({drug['generic_name']})
Category: {drug['search_category']} | Type: {drug['product_type']}
Indications: {drug['indications'][:600]}
Warnings: {drug['warnings'][:400]}
Interactions: {drug['drug_interactions'][:400]}
Adverse Reactions: {drug['adverse_reactions'][:300]}""".strip()

                    all_drugs.append(drug)
                    category_count += 1

                print(f"  {category}: {category_count}/{PER_CATEGORY_LIMIT} valid | Total: {len(all_drugs)}")
                skip += limit
                time.sleep(0.3)

            except Exception as e:
                print(f"  Error: {e}")
                break

        print(f"  Finished {category}: {category_count} valid records")

    df = pd.DataFrame(all_drugs)
    before = len(df)
    df = df.drop_duplicates(subset=["set_id"]).reset_index(drop=True)
    after = len(df)
    print(f"\nDeduplication: {before} → {after} unique records")
    print("\nRecords per category:")
    print(df["search_category"].value_counts().to_string())
    return df


if __name__ == "__main__":
    df = fetch_all_drug_labels()
    df.to_csv("data/drug_labels_full.csv", index=False)
    print(f"\nSaved {len(df)} records to data/drug_labels_full.csv")
    print(df[["brand_name", "generic_name", "search_category", "has_black_box"]].head(10))