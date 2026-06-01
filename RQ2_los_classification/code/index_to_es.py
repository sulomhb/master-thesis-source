"""Bulk-index train and test JSONL files into Elasticsearch with Norwegian analyzer."""
import json
import sys
import requests
from pathlib import Path

ES_URL = "http://localhost:9200"
DATA_DIR = Path(__file__).parent / "data"

INDICES = {
    "los-documents-train": DATA_DIR / "dataset_train.jsonl",
    "los-documents-test": DATA_DIR / "dataset_test.jsonl",
}

MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "norwegian_analyzer": {
                    "type": "norwegian"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "url": {"type": "keyword"},
            "tittel": {"type": "text", "analyzer": "norwegian_analyzer"},
            "innhold": {"type": "text", "analyzer": "norwegian_analyzer", "fielddata": True},
            "innholdsprodusent": {"type": "keyword"},
            "s3_key": {"type": "keyword"},
            "nivaa1": {"type": "text", "analyzer": "norwegian_analyzer", "fields": {"keyword": {"type": "keyword"}}},
            "nivaa2": {"type": "text", "analyzer": "norwegian_analyzer", "fields": {"keyword": {"type": "keyword"}}},
            "nivaa3": {"type": "text", "analyzer": "norwegian_analyzer", "fields": {"keyword": {"type": "keyword"}}},
            "split": {"type": "keyword"},
        }
    }
}


def normalize_record(record, split_value):
    """Map Norwegian field names with å to ASCII variants used by ES."""
    return {
        "url": record.get("url"),
        "tittel": record.get("tittel"),
        "innhold": record.get("innhold"),
        "innholdsprodusent": record.get("innholdsprodusent"),
        "s3_key": record.get("s3_key"),
        "nivaa1": record.get("nivå 1") or record.get("nivaa1"),
        "nivaa2": record.get("nivå 2") or record.get("nivaa2"),
        "nivaa3": record.get("nivå 3") or record.get("nivaa3"),
        "split": split_value,
    }


def reset_index(index):
    requests.delete(f"{ES_URL}/{index}")
    r = requests.put(f"{ES_URL}/{index}", json=MAPPING)
    r.raise_for_status()


def bulk_index(index, jsonl_path, split_value):
    print(f"\n=== Indexing {jsonl_path.name} into '{index}' ===")
    reset_index(index)
    bulk_lines = []
    n = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = normalize_record(rec, split_value)
            if not doc["nivaa1"] or doc["nivaa1"] == "NaN":
                # only index docs with a level-1 label
                continue
            bulk_lines.append(json.dumps({"index": {"_index": index, "_id": doc["url"]}}))
            bulk_lines.append(json.dumps(doc, ensure_ascii=False))
            n += 1

    body = "\n".join(bulk_lines) + "\n"
    r = requests.post(
        f"{ES_URL}/_bulk",
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
    )
    r.raise_for_status()
    response = r.json()
    if response.get("errors"):
        # print first 3 errors
        for item in response["items"][:3]:
            print("  ERROR:", item)
        raise SystemExit("bulk index errors")
    print(f"  indexed {n} documents")

    # refresh
    requests.post(f"{ES_URL}/{index}/_refresh")
    count = requests.get(f"{ES_URL}/{index}/_count").json()["count"]
    print(f"  ES reports {count} docs in '{index}'")


def main():
    for index, path in INDICES.items():
        if not path.exists():
            sys.exit(f"missing {path}")
        bulk_index(index, path, split_value="train" if "train" in index else "test")

    # quick categories sanity
    for index in INDICES:
        agg = {
            "size": 0,
            "aggs": {"cats": {"terms": {"field": "nivaa1.keyword", "size": 10}}}
        }
        r = requests.post(f"{ES_URL}/{index}/_search", json=agg).json()
        buckets = r["aggregations"]["cats"]["buckets"]
        print(f"\n{index} categories:")
        for b in buckets:
            print(f"  {b['key']:32s} {b['doc_count']}")


if __name__ == "__main__":
    main()
