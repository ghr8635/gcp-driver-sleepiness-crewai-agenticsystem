"""
Create BigQuery dataset/tables and optionally upload raw data to Cloud Storage.

Usage:
    python app/setup_gcp_resources.py \
      --project YOUR_PROJECT \
      --dataset driver_sleepiness_ai \
      --location EU \
      --bucket YOUR_BUCKET \
      --upload_raw
"""
import argparse
from pathlib import Path
from google.cloud import bigquery, storage


def load_schema(path):
    import json
    fields = json.loads(Path(path).read_text(encoding="utf-8"))
    return [bigquery.SchemaField(f["name"], f["type"], mode=f.get("mode", "NULLABLE")) for f in fields]


def create_dataset(client, project, dataset, location):
    dataset_id = f"{project}.{dataset}"
    ds = bigquery.Dataset(dataset_id)
    ds.location = location
    return client.create_dataset(ds, exists_ok=True)


def create_table(client, table_id, schema_path):
    table = bigquery.Table(table_id, schema=load_schema(schema_path))
    return client.create_table(table, exists_ok=True)


def create_bucket_if_missing(storage_client, bucket_name, location):
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        bucket = storage_client.create_bucket(bucket_name, location=location)
    return bucket


def upload_file(bucket, local_path, blob_name):
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket.name}/{blob_name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", default="driver_sleepiness_ai")
    parser.add_argument("--location", default="EU")
    parser.add_argument("--bucket", required=False)
    parser.add_argument("--upload_raw", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    bq_client = bigquery.Client(project=args.project)

    create_dataset(bq_client, args.project, args.dataset, args.location)
    create_table(
        bq_client,
        f"{args.project}.{args.dataset}.fatigue_features",
        root / "schemas/fatigue_features_schema.json",
    )
    create_table(
        bq_client,
        f"{args.project}.{args.dataset}.agent_decision_logs",
        root / "schemas/agent_decision_logs_schema.json",
    )
    print(f"Created/verified BigQuery dataset and tables: {args.project}.{args.dataset}")

    if args.bucket and args.upload_raw:
        storage_client = storage.Client(project=args.project)
        bucket = create_bucket_if_missing(storage_client, args.bucket, args.location)
        uri = upload_file(
            bucket,
            root / "data/raw_synced_driver_events.jsonl",
            "raw/raw_synced_driver_events.jsonl",
        )
        print(f"Uploaded raw data to: {uri}")


if __name__ == "__main__":
    main()
