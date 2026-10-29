"""Dataset Loading, JSON Conversion, and MongoDB Import Utility.

Supports reading from CSV and JSON, adding consistent record identifiers,
exporting JSON documents, and idempotent batch importing into MongoDB.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import urllib.request
import urllib.error
import pandas as pd
import pymongo
from pymongo import UpdateOne
from pymongo.errors import PyMongoError

from src.config import (
    RAW_CSV_PATH,
    RAW_JSON_PATH,
    DATASET_URL,
    logger,
)
from src.database import get_collection, init_db


def download_dataset_if_missing(
    url: str = DATASET_URL,
    target_path: Optional[Path] = None,
) -> Path:
    """Download the Medical Cost Personal Dataset if it does not already exist.

    Args:
        url: Remote raw CSV URL.
        target_path: Local destination Path. Defaults to RAW_CSV_PATH.

    Returns:
        Path to the downloaded or existing CSV file.

    Raises:
        FileNotFoundError: If download fails and file is not present locally.
    """
    path = target_path or RAW_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        logger.info(f"Dataset already exists at: {path}")
        return path

    logger.info(f"Downloading insurance dataset from {url}...")
    try:
        # User-agent header to avoid bot filtering
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=15) as response, open(path, "wb") as out_file:
            out_file.write(response.read())
        logger.info(f"Dataset successfully downloaded to {path}")
        return path
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        msg = (
            f"Failed to automatically download dataset from {url}. Reason: {exc}\n"
            f"MANUAL SETUP INSTRUCTIONS:\n"
            f"1. Download the insurance.csv file directly from:\n"
            f"   {url}\n"
            f"2. Place the file at: {path.resolve()}\n"
            f"3. Rerun the command."
        )
        logger.error(msg)
        raise FileNotFoundError(msg) from exc


def load_csv_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load dataset from a CSV file.

    Args:
        filepath: Path to the CSV file. Defaults to RAW_CSV_PATH.

    Returns:
        Pandas DataFrame with raw data.
    """
    path = Path(filepath or RAW_CSV_PATH)
    if not path.exists():
        path = download_dataset_if_missing(target_path=path)

    df = pd.read_csv(path)
    logger.info(f"Loaded CSV dataset from {path} with shape {df.shape}")
    return df


def ensure_record_identifiers(df: pd.DataFrame, prefix: str = "INS") -> pd.DataFrame:
    """Ensure every row has a unique, deterministic 'record_id' column.

    Args:
        df: Input DataFrame.
        prefix: Prefix for formatted identifier string.

    Returns:
        DataFrame with 'record_id' as the first column.
    """
    df_copy = df.copy()
    if "record_id" not in df_copy.columns:
        # Create zero-padded deterministic IDs like INS_00001, INS_00002
        df_copy.insert(0, "record_id", [f"{prefix}_{i+1:05d}" for i in range(len(df_copy))])
    return df_copy


def export_dataframe_to_json(
    df: pd.DataFrame,
    json_path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> Path:
    """Export DataFrame records to formatted JSON file with unique record identifiers.

    Args:
        df: Input DataFrame.
        json_path: Path where JSON will be saved. Defaults to RAW_JSON_PATH.
        indent: JSON indentation formatting.

    Returns:
        Path to the saved JSON file.
    """
    target = Path(json_path or RAW_JSON_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)

    df_with_id = ensure_record_identifiers(df)
    records = df_with_id.to_dict(orient="records")

    with open(target, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=indent)

    logger.info(f"Exported {len(records)} records to JSON at: {target}")
    return target


def load_json_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load dataset from a JSON file.

    Args:
        filepath: Path to JSON file. Defaults to RAW_JSON_PATH.

    Returns:
        Pandas DataFrame loaded from JSON.
    """
    path = Path(filepath or RAW_JSON_PATH)
    if not path.exists():
        # Generate JSON from CSV if JSON doesn't exist yet
        csv_df = load_csv_data()
        export_dataframe_to_json(csv_df, json_path=path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"Loaded JSON dataset from {path} with shape {df.shape}")
    return df


def import_to_mongodb(
    data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
    collection: Optional[pymongo.collection.Collection] = None,
    upsert: bool = True,
) -> Dict[str, Any]:
    """Import dataset into MongoDB using PyMongo without duplicating existing records.

    Uses MongoDB bulk write with upsert on 'record_id' to ensure idempotency.

    Args:
        data: DataFrame or list of dict records. If None, loads CSV/JSON automatically.
        collection: Target PyMongo collection. If None, gets default from database.py.
        upsert: If True, updates existing records and inserts new ones without duplicate errors.

    Returns:
        Dictionary summary containing processed, inserted/upserted, and matched counts.
    """
    coll = collection if collection is not None else get_collection()
    # Ensure index exists on record_id
    init_db(coll_name=coll.name)

    if data is None:
        if RAW_JSON_PATH.exists():
            df = load_json_data(RAW_JSON_PATH)
        else:
            df = load_csv_data(RAW_CSV_PATH)
            export_dataframe_to_json(df, RAW_JSON_PATH)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    df_ready = ensure_record_identifiers(df)
    records = df_ready.to_dict(orient="records")

    if not records:
        logger.warning("No records to import.")
        return {"total_records": 0, "upserted": 0, "matched": 0, "modified": 0}

    # Prepare bulk upsert operations to prevent duplicate primary keys
    operations = []
    for rec in records:
        # Remove MongoDB _id if present in incoming data to avoid immutable field errors
        rec_clean = {k: v for k, v in rec.items() if k != "_id"}
        operations.append(
            UpdateOne(
                {"record_id": rec_clean["record_id"]},
                {"$set": rec_clean},
                upsert=upsert,
            )
        )

    try:
        bulk_result = coll.bulk_write(operations, ordered=False)
        summary = {
            "total_records": len(records),
            "matched_count": bulk_result.matched_count,
            "modified_count": bulk_result.modified_count,
            "upserted_count": len(bulk_result.upserted_ids),
            "collection_total": coll.count_documents({}),
        }
        logger.info(
            f"MongoDB Import Complete: {summary['upserted_count']} upserted, "
            f"{summary['matched_count']} matched, "
            f"Collection count: {summary['collection_total']}"
        )
        return summary
    except PyMongoError as exc:
        logger.error(f"Error during bulk import to MongoDB: {exc}")
        raise


def reset_collection(collection: Optional[pymongo.collection.Collection] = None) -> int:
    """Safely reset (delete all documents from) the target collection for maintenance/testing.

    WARNING: Only run when explicitly requested by user or test scripts.

    Args:
        collection: Target collection.

    Returns:
        Number of documents deleted.
    """
    coll = collection if collection is not None else get_collection()
    result = coll.delete_many({})
    logger.info(f"Reset collection '{coll.name}'. Deleted {result.deleted_count} documents.")
    # Re-initialize index after reset
    init_db(coll_name=coll.name)
    return result.deleted_count
