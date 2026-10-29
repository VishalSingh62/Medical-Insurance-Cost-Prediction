"""MongoDB and PyMongo Database Connectivity Module.

Handles MongoDB client initialization, connection verification via ping,
index creation on unique identifiers, and connection lifecycle management.
"""

from typing import Optional, Tuple
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError

from src.config import (
    MONGODB_URI,
    DATABASE_NAME,
    COLLECTION_NAME,
    MONGO_TIMEOUT_MS,
    logger,
)


def get_mongo_client(
    uri: Optional[str] = None,
    timeout_ms: int = MONGO_TIMEOUT_MS,
) -> pymongo.MongoClient:
    """Create and return a PyMongo MongoClient instance.

    Args:
        uri: MongoDB connection URI string. Defaults to MONGODB_URI from config.
        timeout_ms: Server selection timeout in milliseconds.

    Returns:
        pymongo.MongoClient instance.
    """
    connection_uri = uri or MONGODB_URI
    return pymongo.MongoClient(
        connection_uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
    )


def check_connection(
    client: Optional[pymongo.MongoClient] = None,
    uri: Optional[str] = None,
) -> Tuple[bool, str]:
    """Verify connectivity to MongoDB using the admin 'ping' command.

    Args:
        client: Existing MongoClient instance. If None, creates a temporary one.
        uri: Optional connection URI if client is created.

    Returns:
        Tuple of (is_connected: bool, message: str)
    """
    should_close = False
    if client is None:
        client = get_mongo_client(uri=uri)
        should_close = True

    try:
        # The ping command is lightweight and confirms cluster responsiveness
        client.admin.command("ping")
        return True, "Successfully connected to MongoDB server."
    except ServerSelectionTimeoutError as exc:
        msg = (
            f"MongoDB server selection timed out. Please verify that MongoDB Community Server "
            f"is installed, running on localhost:27017, or that your MONGODB_URI is correct. "
            f"Details: {exc}"
        )
        logger.warning(msg)
        return False, msg
    except ConnectionFailure as exc:
        msg = f"MongoDB connection failure. Details: {exc}"
        logger.warning(msg)
        return False, msg
    except PyMongoError as exc:
        msg = f"PyMongo error occurred while testing connection: {exc}"
        logger.warning(msg)
        return False, msg
    except Exception as exc:
        msg = f"Unexpected error during MongoDB connection check: {exc}"
        logger.error(msg)
        return False, msg
    finally:
        if should_close:
            try:
                client.close()
            except Exception:
                pass


def get_database(
    client: Optional[pymongo.MongoClient] = None,
    db_name: Optional[str] = None,
) -> pymongo.database.Database:
    """Retrieve the target database instance.

    Args:
        client: Active MongoClient instance. If None, initializes a new client.
        db_name: Database name. Defaults to DATABASE_NAME from config.

    Returns:
        pymongo Database instance.
    """
    active_client = client or get_mongo_client()
    target_db_name = db_name or DATABASE_NAME
    return active_client[target_db_name]


def get_collection(
    client: Optional[pymongo.MongoClient] = None,
    db_name: Optional[str] = None,
    coll_name: Optional[str] = None,
) -> pymongo.collection.Collection:
    """Retrieve the target collection instance.

    Args:
        client: Active MongoClient instance.
        db_name: Target database name.
        coll_name: Target collection name. Defaults to COLLECTION_NAME.

    Returns:
        pymongo Collection instance.
    """
    db = get_database(client=client, db_name=db_name)
    target_coll_name = coll_name or COLLECTION_NAME
    return db[target_coll_name]


def init_db(
    client: Optional[pymongo.MongoClient] = None,
    db_name: Optional[str] = None,
    coll_name: Optional[str] = None,
) -> bool:
    """Initialize database and ensure unique index on 'record_id'.

    Args:
        client: Active MongoClient.
        db_name: Database name.
        coll_name: Collection name.

    Returns:
        True if index creation succeeded, False otherwise.
    """
    try:
        coll = get_collection(client=client, db_name=db_name, coll_name=coll_name)
        # Create a unique index on record_id to avoid duplicate imports
        coll.create_index([("record_id", pymongo.ASCENDING)], unique=True, name="idx_record_id_unique")
        logger.info("Unique index on 'record_id' successfully verified or created.")
        return True
    except PyMongoError as exc:
        logger.error(f"Failed to create unique index on collection: {exc}")
        return False


def close_connection(client: Optional[pymongo.MongoClient]) -> None:
    """Safely close a MongoClient instance.

    Args:
        client: MongoClient instance to close.
    """
    if client is not None:
        try:
            client.close()
            logger.debug("MongoDB connection closed.")
        except Exception as exc:
            logger.warning(f"Error while closing MongoDB client: {exc}")
