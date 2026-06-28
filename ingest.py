"""
ingest.py — Pull complaints from CFPB API.
Confirmed working structure: list of {_index, _id, _score, _source, sort}
The actual complaint data lives inside _source.
"""

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.consumerfinance.gov/data-research/consumer-complaints/search/",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
})


def warm_up_session():
    print("Warming up session...")
    resp = SESSION.get(
        "https://www.consumerfinance.gov/data-research/consumer-complaints/search/",
        timeout=15
    )
    print(f"  Search page: {resp.status_code}")
    time.sleep(1)


def fetch_window(date_min: str, date_max: str, size: int = 10000) -> list:
    params = {
        "size": size,
        "has_narrative": "true",
        "date_received_min": date_min,
        "date_received_max": date_max,
        "sort": "created_date_desc",
        "format": "json",
    }
    resp = SESSION.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # API returns a flat list of ES hits — unpack _source from each
    if isinstance(data, list):
        records = [item["_source"] for item in data if "_source" in item]
    else:
        records = []
        print(f"  Unexpected response type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

    print(f"  {date_min} → {date_max}: {len(records)} records")
    return records


def fetch_complaints(total_rows: int = 10_000) -> pd.DataFrame:
    """Slide 90-day windows backward until we have enough rows."""
    warm_up_session()

    all_records = []
    end_date = datetime.today()
    attempts = 0

    while len(all_records) < total_rows and attempts < 12:
        start_date = end_date - timedelta(days=90)
        date_min = start_date.strftime("%Y-%m-%d")
        date_max = end_date.strftime("%Y-%m-%d")

        try:
            records = fetch_window(date_min, date_max,
                                   size=min(total_rows - len(all_records), 10000))
            all_records.extend(records)
            print(f"  Running total: {len(all_records):,}")
            if len(records) == 0:
                print("  Empty window, sliding back...")
        except requests.HTTPError as e:
            print(f"  ❌ HTTP {e.response.status_code}: {e}")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback; traceback.print_exc()
            break

        if len(all_records) >= total_rows:
            break

        end_date = start_date - timedelta(days=1)
        attempts += 1
        time.sleep(0.5)

    if not all_records:
        raise RuntimeError("No records fetched.")

    df = pd.DataFrame(all_records)

    # Rename narrative column to standard name
    if "complaint_what_happened" in df.columns:
        df = df.rename(columns={"complaint_what_happened": "consumer_complaint_narrative"})

    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    return df.head(total_rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    args = parser.parse_args()

    print(f"Fetching up to {args.rows:,} complaints...\n")
    df = fetch_complaints(total_rows=args.rows)

    df.to_parquet(DATA_DIR / "complaints.parquet", index=False)
    df.to_csv(DATA_DIR / "complaints.csv", index=False)

    print(f"\n✅ Saved {len(df):,} rows to data/complaints.parquet + data/complaints.csv")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample:")
    print(df[["date_received", "product", "issue", "consumer_complaint_narrative"]].head(3).to_string())
