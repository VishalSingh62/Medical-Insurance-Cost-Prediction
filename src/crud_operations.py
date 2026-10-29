"""MongoDB CRUD Operations and Demonstration Module.

Implements all four database operations (Create, Read, Update, Delete)
with safety guards, parameter validation, and an idempotent demonstration runner.
"""

from typing import Dict, Any, List, Optional, Tuple
import pymongo
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, DuplicateKeyError

from src.config import logger
from src.database import get_collection, init_db
from src.preprocessing import validate_single_record


# ---------------------------------------------------------
# 1. CREATE Operations
# ---------------------------------------------------------

def create_single_record(
    collection: Optional[Collection],
    record_data: Dict[str, Any],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Insert a single insurance document into MongoDB.

    Validates schema and uniqueness of 'record_id'.

    Args:
        collection: PyMongo Collection instance.
        record_data: Document dictionary to insert.

    Returns:
        Tuple of (success: bool, message: str, inserted_doc: Optional[dict])
    """
    coll = collection if collection is not None else get_collection()

    # Validate schema
    is_valid, err_msg = validate_single_record(record_data)
    if not is_valid:
        return False, f"Validation failed: {err_msg}", None

    clean_record = {k: v for k, v in record_data.items() if k != "_id"}
    if "record_id" not in clean_record:
        import uuid
        clean_record["record_id"] = f"INS_CUSTOM_{uuid.uuid4().hex[:6].upper()}"

    try:
        result = coll.insert_one(clean_record)
        clean_record["_id"] = str(result.inserted_id)
        logger.info(f"Inserted record with record_id '{clean_record['record_id']}'")
        return True, f"Record '{clean_record['record_id']}' successfully created.", clean_record
    except DuplicateKeyError:
        msg = f"Record with record_id '{clean_record['record_id']}' already exists."
        logger.warning(msg)
        return False, msg, None
    except PyMongoError as exc:
        msg = f"Database error during insertion: {exc}"
        logger.error(msg)
        return False, msg, None


def create_batch_records(
    collection: Optional[Collection],
    records_list: List[Dict[str, Any]],
) -> Tuple[int, List[str]]:
    """Insert a batch of documents into MongoDB.

    Args:
        collection: PyMongo Collection.
        records_list: List of document dictionaries.

    Returns:
        Tuple of (inserted_count: int, inserted_record_ids: List[str])
    """
    coll = collection if collection is not None else get_collection()
    if not records_list:
        return 0, []

    valid_docs = []
    for doc in records_list:
        clean = {k: v for k, v in doc.items() if k != "_id"}
        is_valid, _ = validate_single_record(clean)
        if is_valid:
            valid_docs.append(clean)

    if not valid_docs:
        return 0, []

    try:
        # ordered=False allows valid docs to insert even if some IDs conflict
        res = coll.insert_many(valid_docs, ordered=False)
        inserted_ids = [d["record_id"] for d in valid_docs if "record_id" in d]
        logger.info(f"Batch inserted {len(res.inserted_ids)} records.")
        return len(res.inserted_ids), inserted_ids
    except PyMongoError as exc:
        logger.warning(f"Partial batch insertion with notice: {exc}")
        return 0, []


# ---------------------------------------------------------
# 2. READ Operations
# ---------------------------------------------------------

def read_record_by_id(
    collection: Optional[Collection],
    record_id: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve a single document by its business record identifier ('record_id').

    Args:
        collection: PyMongo Collection.
        record_id: Identifier string (e.g., 'INS_00001').

    Returns:
        Document dict if found, None otherwise.
    """
    coll = collection if collection is not None else get_collection()
    doc = coll.find_one({"record_id": record_id}, {"_id": 0})
    return doc


def read_all_records(
    collection: Optional[Collection],
    limit: int = 100,
    skip: int = 0,
) -> List[Dict[str, Any]]:
    """Retrieve all records with pagination.

    Args:
        collection: PyMongo Collection.
        limit: Maximum documents to return.
        skip: Number of documents to skip.

    Returns:
        List of document dictionaries.
    """
    coll = collection if collection is not None else get_collection()
    cursor = coll.find({}, {"_id": 0}).skip(skip).limit(limit)
    return list(cursor)


def filter_records(
    collection: Optional[Collection],
    query: Optional[Dict[str, Any]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Filter records based on a MongoDB query document.

    Args:
        collection: PyMongo Collection.
        query: MongoDB filter dict (e.g. {'smoker': 'yes', 'age': {'$gte': 40}}).
        limit: Maximum results to return.

    Returns:
        List of matching document dictionaries.
    """
    coll = collection if collection is not None else get_collection()
    filter_dict = query or {}
    cursor = coll.find(filter_dict, {"_id": 0}).limit(limit)
    return list(cursor)


# ---------------------------------------------------------
# 3. UPDATE Operations
# ---------------------------------------------------------

def update_record(
    collection: Optional[Collection],
    record_id: str,
    update_fields: Dict[str, Any],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Modify an existing record identified by 'record_id'.

    Prevents updating the immutable 'record_id' and '_id'.

    Args:
        collection: PyMongo Collection.
        record_id: Unique record ID to update.
        update_fields: Fields to update.

    Returns:
        Tuple of (success: bool, message: str, updated_doc: Optional[dict])
    """
    coll = collection if collection is not None else get_collection()

    clean_updates = {k: v for k, v in update_fields.items() if k not in ("_id", "record_id")}
    if not clean_updates:
        return False, "No valid update fields provided.", None

    try:
        result = coll.update_one(
            {"record_id": record_id},
            {"$set": clean_updates},
        )

        if result.matched_count == 0:
            return False, f"Record with record_id '{record_id}' not found.", None

        updated_doc = coll.find_one({"record_id": record_id}, {"_id": 0})
        logger.info(f"Updated record '{record_id}'. Modified fields: {list(clean_updates.keys())}")
        return True, f"Record '{record_id}' successfully updated.", updated_doc
    except PyMongoError as exc:
        msg = f"Database error during update: {exc}"
        logger.error(msg)
        return False, msg, None


# ---------------------------------------------------------
# 4. DELETE Operations
# ---------------------------------------------------------

def delete_record(
    collection: Optional[Collection],
    record_id: str,
) -> Tuple[bool, str]:
    """Delete a single document identified by 'record_id'.

    Safety guard: prevents deletion if record_id is empty.

    Args:
        collection: PyMongo Collection.
        record_id: Record identifier to delete.

    Returns:
        Tuple of (success: bool, message: str)
    """
    coll = collection if collection is not None else get_collection()

    if not record_id or not str(record_id).strip():
        return False, "Record ID must not be empty."

    try:
        result = coll.delete_one({"record_id": record_id})
        if result.deleted_count == 0:
            return False, f"Record '{record_id}' not found. Nothing deleted."

        logger.info(f"Deleted record with record_id '{record_id}'")
        return True, f"Record '{record_id}' successfully deleted."
    except PyMongoError as exc:
        msg = f"Database error during deletion: {exc}"
        logger.error(msg)
        return False, msg


# ---------------------------------------------------------
# Full CRUD Demonstration Runner
# ---------------------------------------------------------

def run_crud_demonstration(collection: Optional[Collection] = None) -> Dict[str, Any]:
    """Execute a complete, self-cleaning CRUD demonstration.

    This demonstration:
    1. CREATE: Inserts a single test record ('DEMO_TEST_001') and batch records ('DEMO_BATCH_001', 'DEMO_BATCH_002').
    2. READ: Retrieves the record by ID and performs a filtered query.
    3. UPDATE: Updates BMI and charges for 'DEMO_TEST_001' and verifies changes.
    4. DELETE: Deletes all demo records so no duplicate residue is left.

    Returns:
        Structured dictionary summary of all test operations.
    """
    coll = collection if collection is not None else get_collection()
    init_db(coll_name=coll.name)

    demo_id = "DEMO_TEST_001"
    batch_ids = ["DEMO_BATCH_001", "DEMO_BATCH_002"]
    all_demo_ids = [demo_id] + batch_ids

    # Step 0: Pre-clean in case a previous interrupted run left demo documents
    coll.delete_many({"record_id": {"$in": all_demo_ids}})

    results: Dict[str, Any] = {}

    print("\n" + "=" * 60)
    print("EXECUTING PYMONGO CRUD DEMONSTRATION")
    print("=" * 60)


    # 1. CREATE: Single Insert
    single_doc = {
        "record_id": demo_id,
        "age": 32,
        "sex": "female",
        "bmi": 26.5,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
        "charges": 4500.0,
    }
    s_ok, s_msg, s_doc = create_single_record(coll, single_doc)
    print(f"\n[1. CREATE - Single Record]: {s_msg}")
    results["create_single"] = {"success": s_ok, "record": s_doc}

    # 1b. CREATE: Batch Insert
    batch_docs = [
        {
            "record_id": "DEMO_BATCH_001",
            "age": 45,
            "sex": "male",
            "bmi": 31.2,
            "children": 2,
            "smoker": "yes",
            "region": "southeast",
            "charges": 28500.0,
        },
        {
            "record_id": "DEMO_BATCH_002",
            "age": 22,
            "sex": "female",
            "bmi": 23.8,
            "children": 0,
            "smoker": "no",
            "region": "southwest",
            "charges": 2200.0,
        },
    ]
    b_count, b_ids = create_batch_records(coll, batch_docs)
    print(f"[1b. CREATE - Batch Insert]: Inserted {b_count} records ({b_ids})")
    results["create_batch"] = {"count": b_count, "ids": b_ids}

    # 2. READ: Read by ID
    read_doc = read_record_by_id(coll, demo_id)
    print(f"\n[2. READ - By ID '{demo_id}']:\n    {read_doc}")
    results["read_by_id"] = read_doc

    # 2b. READ: Filter query (smoker == 'yes')
    filtered = filter_records(coll, {"record_id": {"$in": all_demo_ids}, "smoker": "yes"})
    print(f"[2b. READ - Filtered Demo Records (smoker=yes)]: Found {len(filtered)} match(es)")
    results["read_filtered"] = filtered

    # 3. UPDATE: Modify test record
    update_data = {"bmi": 29.8, "charges": 5200.0}
    u_ok, u_msg, u_doc = update_record(coll, demo_id, update_data)
    print(f"\n[3. UPDATE - Modify '{demo_id}']:\n    Status: {u_msg}\n    Updated Values: BMI={u_doc.get('bmi')}, Charges={u_doc.get('charges')}")
    results["update"] = {"success": u_ok, "updated_doc": u_doc}

    # 4. DELETE: Clean up test records
    print(f"\n[4. DELETE - Cleaning Test Records]:")
    del_results = []
    for test_id in all_demo_ids:
        d_ok, d_msg = delete_record(coll, test_id)
        print(f"    - {d_msg}")
        del_results.append({"record_id": test_id, "deleted": d_ok})
    results["delete"] = del_results

    # Verification: Confirm test record no longer exists
    remaining = coll.count_documents({"record_id": {"$in": all_demo_ids}})
    print(f"\n[Verification]: Remaining demo records in database = {remaining}")
    results["cleanup_verified"] = (remaining == 0)

    print("=" * 60)
    print("[SUCCESS] CRUD DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")

    return results
