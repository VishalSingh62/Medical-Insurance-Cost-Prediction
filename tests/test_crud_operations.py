"""Unit tests for MongoDB CRUD operations using mongomock.

Isolates tests completely from production database and validates all CRUD functions.
"""

import pytest
import mongomock

from src.crud_operations import (
    create_single_record,
    create_batch_records,
    read_record_by_id,
    read_all_records,
    filter_records,
    update_record,
    delete_record,
    run_crud_demonstration,
)


@pytest.fixture
def mock_collection():
    """Provide an isolated mock MongoDB collection for testing."""
    client = mongomock.MongoClient()
    db = client["test_medical_insurance_db"]
    coll = db["test_insurance_records"]
    coll.create_index([("record_id", 1)], unique=True)
    return coll


def test_create_single_record_success(mock_collection):
    """Test successful insertion of a single document."""
    doc = {
        "record_id": "TEST_001",
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "smoker": "no",
        "region": "southwest",
        "charges": 4000.0,
    }
    success, msg, inserted = create_single_record(mock_collection, doc)
    assert success is True
    assert inserted is not None
    assert inserted["record_id"] == "TEST_001"
    assert mock_collection.count_documents({}) == 1


def test_create_single_record_duplicate_prevention(mock_collection):
    """Test that inserting duplicate record_id is rejected."""
    doc = {
        "record_id": "TEST_DUP",
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "smoker": "no",
        "region": "southwest",
        "charges": 4000.0,
    }
    success1, _, _ = create_single_record(mock_collection, doc)
    assert success1 is True

    # Attempt to insert same record_id again
    success2, msg2, _ = create_single_record(mock_collection, doc)
    assert success2 is False
    assert "already exists" in msg2


def test_create_single_record_validation_failure(mock_collection):
    """Test that invalid documents are rejected before database insertion."""
    bad_doc = {
        "record_id": "TEST_BAD",
        "age": -5,  # Invalid
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "smoker": "no",
        "region": "southwest",
    }
    success, msg, _ = create_single_record(mock_collection, bad_doc)
    assert success is False
    assert "Validation failed" in msg


def test_create_batch_records(mock_collection):
    """Test inserting multiple documents in batch."""
    docs = [
        {"record_id": "BATCH_01", "age": 22, "sex": "female", "bmi": 21.0, "children": 0, "smoker": "no", "region": "northeast"},
        {"record_id": "BATCH_02", "age": 44, "sex": "male", "bmi": 29.5, "children": 2, "smoker": "yes", "region": "southeast"},
    ]
    count, ids = create_batch_records(mock_collection, docs)
    assert count == 2
    assert "BATCH_01" in ids
    assert "BATCH_02" in ids
    assert mock_collection.count_documents({}) == 2


def test_read_operations(mock_collection):
    """Test read by ID, read all, and filtered reads."""
    mock_collection.insert_many(
        [
            {"record_id": "R_01", "age": 20, "sex": "female", "bmi": 22.0, "children": 0, "smoker": "no", "region": "northwest"},
            {"record_id": "R_02", "age": 55, "sex": "male", "bmi": 33.0, "children": 3, "smoker": "yes", "region": "southeast"},
        ]
    )

    # Read by ID
    doc1 = read_record_by_id(mock_collection, "R_01")
    assert doc1 is not None
    assert doc1["age"] == 20

    # Read non-existent
    assert read_record_by_id(mock_collection, "NON_EXISTENT") is None

    # Read all
    all_docs = read_all_records(mock_collection)
    assert len(all_docs) == 2

    # Filtered
    smokers = filter_records(mock_collection, {"smoker": "yes"})
    assert len(smokers) == 1
    assert smokers[0]["record_id"] == "R_02"


def test_update_record(mock_collection):
    """Test updating existing document fields."""
    mock_collection.insert_one(
        {"record_id": "UP_01", "age": 30, "sex": "male", "bmi": 24.0, "children": 0, "smoker": "no", "region": "southwest"}
    )

    success, msg, updated = update_record(mock_collection, "UP_01", {"bmi": 26.5, "children": 1})
    assert success is True
    assert updated["bmi"] == 26.5
    assert updated["children"] == 1

    # Verify in collection
    stored = mock_collection.find_one({"record_id": "UP_01"})
    assert stored["bmi"] == 26.5


def test_delete_record(mock_collection):
    """Test deleting document by record_id."""
    mock_collection.insert_one(
        {"record_id": "DEL_01", "age": 30, "sex": "male", "bmi": 24.0, "children": 0, "smoker": "no", "region": "southwest"}
    )
    assert mock_collection.count_documents({}) == 1

    success, msg = delete_record(mock_collection, "DEL_01")
    assert success is True
    assert mock_collection.count_documents({}) == 0

    # Deleting again should return False
    success_retry, _ = delete_record(mock_collection, "DEL_01")
    assert success_retry is False


def test_run_crud_demonstration_on_mock(mock_collection):
    """Test that the full CRUD demonstration runs end-to-end and cleans up."""
    demo_results = run_crud_demonstration(collection=mock_collection)
    assert demo_results["create_single"]["success"] is True
    assert demo_results["create_batch"]["count"] == 2
    assert demo_results["update"]["success"] is True
    assert demo_results["cleanup_verified"] is True
