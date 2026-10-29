"""Command-Line Interface (CLI) Entrypoint for Medical Insurance ML Project.

Provides commands to import data into MongoDB, demonstrate CRUD operations,
train and evaluate regression models, generate visualizations, and run sample inference.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import logger
from src.database import check_connection, get_collection
from src.data_loader import (
    download_dataset_if_missing,
    load_csv_data,
    export_dataframe_to_json,
    import_to_mongodb,
)
from src.preprocessing import clean_and_preprocess_dataset
from src.crud_operations import run_crud_demonstration
from src.train_models import train_and_evaluate_all_models
from src.visualizations import generate_all_visualizations
from src.prediction import predict_insurance_charge


def handle_check_db(args: argparse.Namespace) -> None:
    """Check MongoDB connectivity and display server status."""
    print("\n--- MongoDB Connection Diagnostic ---")
    connected, msg = check_connection()
    if connected:
        print(f"[OK] Status: CONNECTED\n     {msg}")
        coll = get_collection()
        count = coll.count_documents({})
        print(f"     Database:   {coll.database.name}")
        print(f"     Collection: {coll.name}")
        print(f"     Current Documents: {count}")
    else:
        print(f"[ERROR] Status: DISCONNECTED\n        {msg}")
        print("\nTroubleshooting tips:")
        print("1. Ensure MongoDB Community Server is installed on Windows.")
        print("2. Start the Windows service: net start MongoDB")
        print("3. Or start mongod manually: mongod --dbpath <data-directory>")
    print("-" * 40 + "\n")



def handle_import_data(args: argparse.Namespace) -> None:
    """Download data, preprocess, generate JSON, and import into MongoDB."""
    print("\n--- 1. Dataset Acquisition & Cleaning ---")
    download_dataset_if_missing()
    raw_df = load_csv_data()
    clean_df, summary = clean_and_preprocess_dataset(raw_df, save_cleaned=True)
    print(f"[OK] Initial Records:  {summary['initial_rows']}")
    print(f"[OK] Cleaned Records:  {summary['cleaned_rows']} (Duplicates removed: {summary['duplicates_removed']})")

    print("\n--- 2. JSON Conversion ---")
    json_path = export_dataframe_to_json(clean_df)
    print(f"[OK] Cleaned records exported to JSON: {json_path}")

    print("\n--- 3. MongoDB Idempotent Import ---")
    connected, msg = check_connection()
    if not connected:
        print(f"[WARN] MongoDB is not currently reachable: {msg}")
        print("       Local CSV and JSON files are ready. Start MongoDB and rerun 'python main.py import-data'.")
        return

    import_summary = import_to_mongodb(clean_df)
    print(f"[OK] Total Processed: {import_summary['total_records']}")
    print(f"[OK] Upserted:        {import_summary['upserted_count']}")
    print(f"[OK] Matched:         {import_summary['matched_count']}")
    print(f"[OK] Total in DB:     {import_summary['collection_total']}")
    print("[SUCCESS] Data Import Completed Successfully!\n")


def handle_demo_crud(args: argparse.Namespace) -> None:
    """Execute complete, safe CRUD demonstration on MongoDB."""
    connected, msg = check_connection()
    if not connected:
        print(f"[ERROR] Cannot run CRUD demonstration. MongoDB is unreachable:\n        {msg}")
        sys.exit(1)

    run_crud_demonstration()


def handle_train(args: argparse.Namespace) -> None:
    """Train Linear Regression and Random Forest Regressor models and compare metrics."""
    print("\n--- Training Regression Pipelines ---")
    _, df_comp, best_key = train_and_evaluate_all_models()
    print("[SUCCESS] Model Training & Evaluation Complete!")
    print(f"[BEST MODEL] Selected: {best_key.upper()} (Saved to models/best_model_pipeline.joblib)\n")


def handle_visualize(args: argparse.Namespace) -> None:
    """Generate all 8 analysis and evaluation charts."""
    print("\n--- Generating Visualizations ---")
    charts = generate_all_visualizations()
    print("[SUCCESS] Successfully generated 8 charts:")
    for name, path in charts.items():
        print(f"   - {name:<25} -> {path.name}")
    print("All charts saved to outputs/ directory.\n")


def handle_predict(args: argparse.Namespace) -> None:
    """Predict insurance cost for a specific patient profile."""
    profile = {
        "age": args.age,
        "sex": args.sex,
        "bmi": args.bmi,
        "children": args.children,
        "smoker": args.smoker,
        "region": args.region,
    }

    try:
        res = predict_insurance_charge(profile)
        print("\n" + "=" * 50)
        print("PREDICTED MEDICAL INSURANCE CHARGES")
        print("=" * 50)
        print(f"Patient Profile:")
        print(f"  - Age:      {profile['age']}")
        print(f"  - Sex:      {profile['sex']}")
        print(f"  - BMI:      {profile['bmi']}")
        print(f"  - Children: {profile['children']}")
        print(f"  - Smoker:   {profile['smoker']}")
        print(f"  - Region:   {profile['region']}")
        print("-" * 50)
        print(f"Estimated Annual Charges: {res['predicted_charge_formatted']}")
        print("-" * 50)
        print(f"{res['disclaimer']}")
        print("=" * 50 + "\n")
    except Exception as exc:
        print(f"[ERROR] Prediction failed: {exc}")
        sys.exit(1)


def handle_all(args: argparse.Namespace) -> None:
    """Execute complete pipeline: Import -> CRUD -> Train -> Visualize."""
    print("==================================================")
    print("RUNNING COMPLETE END-TO-END PIPELINE")
    print("==================================================")
    handle_import_data(args)
    handle_demo_crud(args)
    handle_train(args)
    handle_visualize(args)
    print("==================================================")
    print("FULL WORKFLOW FINISHED SUCCESSFULLY!")
    print("To launch the Streamlit web dashboard, run:")
    print("    streamlit run app.py")
    print("==================================================")



def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Medical Insurance ML & MongoDB Analysis System (College CA2 Project)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # check-db
    subparsers.add_parser("check-db", help="Verify MongoDB server connectivity")

    # import-data
    subparsers.add_parser("import-data", help="Download, clean, convert to JSON, and import into MongoDB")

    # demo-crud
    subparsers.add_parser("demo-crud", help="Run safe CRUD operations demonstration on MongoDB")

    # train
    subparsers.add_parser("train", help="Train and compare Linear Regression & Random Forest models")

    # visualize
    subparsers.add_parser("visualize", help="Generate all 8 evaluation and exploratory charts")

    # predict
    pred_parser = subparsers.add_parser("predict", help="Predict charges for a sample patient profile")
    pred_parser.add_argument("--age", type=float, default=35, help="Age in years (18-120)")
    pred_parser.add_argument("--sex", type=str, default="male", choices=["male", "female"], help="Sex")
    pred_parser.add_argument("--bmi", type=float, default=28.5, help="BMI (10-65)")
    pred_parser.add_argument("--children", type=int, default=1, help="Number of dependent children")
    pred_parser.add_argument("--smoker", type=str, default="no", choices=["yes", "no"], help="Smoking status")
    pred_parser.add_argument(
        "--region",
        type=str,
        default="southeast",
        choices=["southwest", "southeast", "northwest", "northeast"],
        help="Residential region",
    )

    # all
    subparsers.add_parser("all", help="Execute entire pipeline: import, crud, train, visualize")

    return parser


def main() -> None:
    """Main CLI routing function."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "check-db": handle_check_db,
        "import-data": handle_import_data,
        "demo-crud": handle_demo_crud,
        "train": handle_train,
        "visualize": handle_visualize,
        "predict": handle_predict,
        "all": handle_all,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
