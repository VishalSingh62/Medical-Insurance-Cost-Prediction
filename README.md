# Medical Insurance Cost Prediction and Analysis Using MongoDB, PyMongo, and Machine Learning

**Course Assessment 2 (CA2) Project**  
**Domain:** Healthcare Analytics, Database Engineering, and Applied Machine Learning  
**Environment:** Python 3.10+, MongoDB Community Server, PyMongo, Scikit-Learn, Streamlit  

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Dataset Source & Attribution](#dataset-source--attribution)
4. [Technology Stack](#technology-stack)
5. [Repository Structure](#repository-structure)
6. [MongoDB Installation & Windows Setup](#mongodb-installation--windows-setup)
7. [Environment Configuration](#environment-configuration)
8. [Installation & Setup](#installation--setup)
9. [Command-Line Workflow (Step-by-Step)](#command-line-workflow-step-by-step)
10. [Machine Learning & Model Evaluation](#machine-learning--model-evaluation)
11. [Why Classification Metrics Are Inapplicable](#why-classification-metrics-are-inapplicable)
12. [Visualizations & Actuarial Findings](#visualizations--actuarial-findings)
13. [Interactive Streamlit Web Dashboard](#interactive-streamlit-web-dashboard)
14. [Testing Suite & Quality Assurance](#testing-suite--quality-assurance)
15. [Troubleshooting Guide](#troubleshooting-guide)
16. [Project Submission Packaging](#project-submission-packaging)

---

## 1. Project Overview

Medical insurance pricing is a critical application of actuarial science and machine learning. Insurance carriers must accurately estimate health expenditures to ensure solvency while maintaining fair, competitive premiums. 

This project implements an end-to-end, production-ready healthcare cost prediction and analysis architecture:
- **Database Layer**: Ingests patient records into **MongoDB Community Server** using **PyMongo**, ensuring data persistence, unique primary index validation, and idempotent bulk upserts.
- **Data Engineering Layer**: Conducts dataset validation, duplicate detection, and Interquartile Range (IQR) outlier reporting without destructive deletion of valid clinical cases.
- **Machine Learning Layer**: Constructs Scikit-Learn **Pipelines** with `ColumnTransformer` preprocessing (one-hot categorical encoding and numerical standardization) to compare **Linear Regression** and **Random Forest Regressor** without target leakage.
- **Presentation Layer**: Deploys an interactive **Streamlit** dashboard featuring live MongoDB CRUD management, model evaluation cards, real-time cost estimation, and exploratory visual analytics.

---

## 2. Objectives

1. **Persistent Data Storage**: Establish a secure local MongoDB database (`medical_insurance_db`) and collection (`insurance_records`) with a unique business index on `record_id`.
2. **Idempotent ETL Pipeline**: Transform raw CSV benchmark data into structured JSON documents and ingest them without accumulating duplicate records.
3. **Comprehensive CRUD Demonstration**: Implement and verify Create, Read, Update, and Delete operations safely without corrupting production data.
4. **Reproducible Machine Learning**: Encapsulate preprocessing and regression algorithms inside Scikit-Learn pipelines using a fixed random seed (`random_state=42`).
5. **Comparative Algorithm Evaluation**: Benchmark Linear Regression against Random Forest Regressor using continuous regression metrics ($R^2$, MAE, RMSE) and select the superior model based strictly on holdout test performance.
6. **Exploratory Visualizations**: Render 8 publication-quality charts detailing cost distributions, feature interactions, and residual distributions.
7. **User-Friendly Dashboard**: Provide a beginner-friendly Streamlit web interface with graceful fallbacks (Demo Mode) when MongoDB is offline.

---

## 3. Dataset Source & Attribution

- **Dataset Name**: Medical Cost Personal Dataset (`insurance.csv`)
- **Author/Curator**: Brett Lantz (*Machine Learning with R*, Packt Publishing)
- **Primary Source Repository**: [stedy/Machine-Learning-with-R-datasets](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)
- **License**: Public Domain (CC0 / Open Data Commons)
- **Dimensions**: 1,338 observations, 7 attributes:
  - `age`: Beneficiary age (18 to 64 years).
  - `sex`: Biological gender (`female`, `male`).
  - `bmi`: Body Mass Index ($\text{kg/m}^2$, normal: 18.5–24.9; obese: $\ge 30$).
  - `children`: Dependent children covered by health insurance (0 to 5).
  - `smoker`: Regular smoking status (`yes`, `no`).
  - `region`: Residential region in the US (`southwest`, `southeast`, `northwest`, `northeast`).
  - `charges`: Billed annual medical expenditures in USD (continuous regression target).

---

## 4. Technology Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10+ / 3.13 | Core runtime environment |
| **Database** | MongoDB Community Server | 8.x / 7.x | NoSQL Document database engine |
| **DB Driver** | PyMongo | $\ge 4.5.0$ | Python connectivity, indexing, CRUD |
| **Data Processing** | Pandas, NumPy | $\ge 2.0.0$ | Data cleaning, reshaping, array operations |
| **Machine Learning**| Scikit-Learn | $\ge 1.3.0$ | `Pipeline`, `ColumnTransformer`, Regression |
| **Model Storage** | Joblib | $\ge 1.3.0$ | Model pipeline serialization |
| **Data Visualization**| Matplotlib, Seaborn | $\ge 3.7.0$ | 8 high-resolution analytical plots |
| **User Interface** | Streamlit | $\ge 1.28.0$ | Multi-tab interactive web dashboard |
| **Unit Testing** | Pytest, mongomock | $\ge 7.4.0$ | Isolated testing and MongoDB mocking |
| **Config / Secrets**| Python-Dotenv | $\ge 1.0.0$ | Environment variable management |

---

## 5. Repository Structure

```
medical-insurance-ml/
├── app.py                      # Interactive Streamlit dashboard application
├── main.py                     # Unified CLI entrypoint (import, crud, train, visualize)
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation and student guide
├── .env.example                # Environment template for MongoDB credentials
├── .env                        # Local active environment configuration (ignored by git)
├── .gitignore                  # Git exclusions (.venv, caches, secrets)
├── data/
│   ├── insurance.csv           # Raw medical cost benchmark dataset
│   ├── insurance.json          # Cleaned records with unique record_id
│   └── insurance_cleaned.csv   # Preprocessed CSV dataset (duplicates removed)
├── src/
│   ├── __init__.py             # Package marker
│   ├── config.py               # Pathlib paths, DB settings, feature definitions
│   ├── database.py             # PyMongo connection, ping health check, index creation
│   ├── data_loader.py          # Download, JSON conversion, idempotent bulk import
│   ├── preprocessing.py        # Data auditing, IQR outliers, ColumnTransformer
│   ├── crud_operations.py      # Safe Create, Read, Update, Delete & demo runner
│   ├── train_models.py         # Linear Regression & Random Forest pipeline training
│   ├── evaluation.py           # Regression metrics (R², MAE, RMSE), comparison table
│   ├── visualizations.py       # 8 publication-quality Matplotlib/Seaborn figures
│   └── prediction.py           # Inference engine, validation & academic disclaimers
├── models/
│   ├── .gitkeep                # Git tracking marker
│   ├── linear_regression_pipeline.joblib   # Trained Linear Regression pipeline
│   ├── random_forest_pipeline.joblib       # Trained Random Forest pipeline
│   └── best_model_pipeline.joblib          # Persisted top-performing pipeline
├── outputs/
│   ├── .gitkeep                # Git tracking marker
│   ├── 01_charges_distribution.png         # Charge distribution & KDE
│   ├── 02_age_vs_charges.png               # Age vs charges with smoker slopes
│   ├── 03_bmi_vs_charges.png               # BMI vs charges (obesity threshold)
│   ├── 04_charges_by_smoker.png            # Box & violin plots by smoker
│   ├── 05_correlation_heatmap.png          # Pearson correlation matrix
│   ├── 06_actual_vs_predicted.png          # Actual vs predicted scatter plots
│   ├── 07_residual_distribution.png        # Residual distributions
│   ├── 08_model_metrics_comparison.png     # R², MAE, RMSE comparison charts
│   ├── model_metrics.json                  # Machine-readable evaluation metrics
│   ├── model_metrics.csv                   # CSV metrics summary
│   ├── best_model_metadata.json            # Top model selection metadata
│   └── preprocessing_summary.json          # Data auditing & IQR outlier summary
└── tests/
    ├── test_preprocessing.py   # Validation, outlier, split & transformer tests
    ├── test_crud_operations.py # Isolated MongoDB CRUD tests via mongomock
    └── test_evaluation.py      # Regression metrics calculation tests
```

---

## 6. MongoDB Installation & Windows Setup

### Option A: MongoDB as a Windows Service (Standard)
1. Download **MongoDB Community Server** MSI installer from [MongoDB Download Center](https://www.mongodb.com/try/download/community).
2. Run the MSI installer and choose **"Complete"** installation.
3. Check the box **"Install MongoDB as a Service"**. This registers the service `MongoDB` running automatically at startup on port `27017`.
4. Verify the service is running in PowerShell:
   ```powershell
   Get-Service -Name "MongoDB"
   ```
   If stopped, start it with administrator privileges:
   ```powershell
   net start MongoDB
   ```

### Option B: Standalone Process Execution
If running without a Windows service:
```powershell
mkdir C:\data\db
"C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe" --dbpath="C:\data\db"
```

---

## 7. Environment Configuration

Copy the configuration template `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Default contents of `.env`:
```ini
# MongoDB Connection URI (Local instance)
MONGODB_URI=mongodb://localhost:27017/

# Target Database and Collection
DATABASE_NAME=medical_insurance_db
COLLECTION_NAME=insurance_records

# Environment
APP_ENV=development
```

---

## 8. Installation & Setup

1. **Clone or navigate** into the project folder:
   ```powershell
   cd medical-insurance-ml
   ```

2. **Create and activate a Python virtual environment**:
   ```powershell
   # Create virtual environment
   py -3 -m venv .venv

   # Activate on Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```
   *(If script execution is restricted on Windows, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 9. Command-Line Workflow (Step-by-Step)

The project includes an intuitive CLI via `main.py` supporting all individual phases and an end-to-end command:

### 1. Test Database Connectivity
```powershell
python main.py check-db
```
*Expected Output:*
```
--- MongoDB Connection Diagnostic ---
[OK] Status: CONNECTED
     Successfully connected to MongoDB server.
     Database:   medical_insurance_db
     Collection: insurance_records
     Current Documents: 1335
----------------------------------------
```

### 2. Ingest Dataset into MongoDB
Downloads raw CSV, validates and removes duplicate records, converts to JSON with unique identifiers (`INS_00001`, `INS_00002`...), and performs an idempotent bulk upsert:
```powershell
python main.py import-data
```

### 3. Run MongoDB CRUD Demonstration
Executes a self-cleaning CRUD cycle (Create single, Create batch, Read by ID, Read filtered, Update, Delete) without modifying real benchmark records:
```powershell
python main.py demo-crud
```

### 4. Train and Compare Machine Learning Models
Trains Scikit-Learn pipelines for Linear Regression and Random Forest Regressor on identical train/test splits (`random_state=42`) and persists models:
```powershell
python main.py train
```

### 5. Generate Visualizations
Renders all 8 analytical and evaluation charts and saves them to `outputs/`:
```powershell
python main.py visualize
```

### 6. Predict Charges for a Patient Profile
```powershell
python main.py predict --age 45 --sex male --bmi 31.5 --children 2 --smoker yes --region southeast
```

### 7. Run Entire Pipeline End-to-End
Runs data import, CRUD demonstration, model training, and visualization generation in a single command:
```powershell
python main.py all
```

---

## 10. Machine Learning & Model Evaluation

Both algorithms are encapsulated in Scikit-Learn `Pipeline` objects paired with a `ColumnTransformer`:
- **Numerical Features** (`age`, `bmi`, `children`): Standardized with `StandardScaler`.
- **Categorical Features** (`sex`, `smoker`, `region`): Encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

### Evaluation Results (Actual Holdout Test Split Metrics)

| Model Architecture | Train $R^2$ | Test $R^2$ | Test MAE ($) | Test RMSE ($) | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.9022** | **0.8886** | **$2,374.09** | **$4,445.15** | **28.32%** |
| **Linear Regression (OLS)** | 0.7414 | 0.7803 | $4,403.74 | $6,243.28 | 43.83% |

### Selection Verdict
The **Random Forest Regressor** is definitively selected as the best-performing model:
- Explains **88.86%** of variance in the test set compared to **78.03%** for Linear Regression.
- Reduces the Root Mean Squared Error (RMSE) by **$1,798.13** (from $6,243.28 down to $4,445.15).
- Achieves a Mean Absolute Error of **$2,374.09**, cutting average deviation nearly in half compared to Linear Regression ($4,403.74).

---

## 11. Why Classification Metrics Are Inapplicable

In academic evaluations, using classification metrics for continuous regression tasks is a serious conceptual error:

| Metric | Valid Task | Mathematical Reason for Inapplicability in Medical Cost Regression |
| :--- | :--- | :--- |
| **Accuracy** | Classification | Measures ratio of exact matches ($\frac{TP+TN}{Total}$). In continuous regression, the probability of predicting the exact floating-point dollar amount (e.g., $16,884.92400) is **0%**. |
| **Precision** | Classification | Requires discrete positive identification ($\frac{TP}{TP+FP}$). Medical charges have no "positive" or "negative" binary classes. |
| **Recall / Sensitivity** | Classification | Measures true positive detection against ground-truth positive classes. Impossible without arbitrary discretization. |
| **F1-Score** | Classification | Harmonic mean of precision and recall. Inherits all classification constraints. |

**The Proper Regression Metrics Used in this Project:**
1. **$R^2$ (Coefficient of Determination)**: Fraction of variance in insurance charges accounted for by predictors ($1 - \frac{SS_{res}}{SS_{tot}}$).
2. **MAE (Mean Absolute Error)**: Average dollar error without squaring: $\frac{1}{n} \sum |y_i - \hat{y}_i|$.
3. **RMSE (Root Mean Squared Error)**: Standard deviation of residuals, penalizing large financial prediction errors: $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$.

---

## 12. Visualizations & Actuarial Findings

All charts are saved at 300 DPI in `outputs/`:

1. **`01_charges_distribution.png`**: Right-skewed distribution of insurance charges, showing a median ($9,382) well below the mean ($13,270) due to high-cost claims.
2. **`02_age_vs_charges.png`**: Shows a steady upward cost trajectory of ~$250–$300 per additional year of age, with an upward vertical intercept shift of ~$20,000 for smokers.
3. **`03_bmi_vs_charges.png`**: Highlights the **clinical obesity threshold (BMI = 30)**. Non-smokers show mild BMI slope; smokers above BMI 30 experience an explosive price jump to $35,000–$45,000+.
4. **`04_charges_by_smoker.png`**: Box and violin plots confirming that smoking is the primary actuarial determinant of high charges.
5. **`05_correlation_heatmap.png`**: Correlation matrix showing `charges` has strongest numerical correlation with `age` ($r = 0.299$) and `bmi` ($r = 0.198$).
6. **`06_actual_vs_predicted.png`**: Scatter plot against the 45° perfect prediction line ($y = x$), demonstrating Random Forest's superior fit across both low and high clusters.
7. **`07_residual_distribution.png`**: Error distribution centered near zero for Random Forest, compared to high residual variance in Linear Regression.
8. **`08_model_metrics_comparison.png`**: Side-by-side bar plots comparing $R^2$, MAE, and RMSE.

---

## 13. Interactive Streamlit Web Dashboard

Launch the interactive dashboard:
```powershell
streamlit run app.py
```

### Dashboard Tabs & Features:
1. **📌 Project Overview**: Live MongoDB connection pill (Online / Offline Demo Mode), architecture diagram, dataset metrics, and feature dictionary.
2. **📊 Data Explorer**: Real-time filters for age, smoker status, region, and gender; summary statistics; IQR outlier reports.
3. **🔄 MongoDB CRUD Operations**: Interactive forms to Insert, Search, Update, and Delete documents with live schema validation.
4. **🤖 Machine Learning & Comparison**: Side-by-side metric tables, comparison bar charts, and simple explanations of OLS and Random Forest.
5. **🔮 Cost Prediction Simulator**: Sliders and selectors to input patient parameters and view estimated charges, along with a calculated **Smoker Surcharge / Non-Smoker Savings** comparison.
6. **📈 Actuarial Insights**: Visual breakdowns of the "Smoker Cliff", age slopes, and residual errors.

---

## 14. Testing Suite & Quality Assurance

Unit tests are implemented using `pytest` and `mongomock` in the `tests/` directory:
- **`tests/test_preprocessing.py`**: Validates domain constraints (age 18–120, BMI 10–65, valid categorical values), IQR outlier detection, duplicate removal, train/test split reproducibility, and transformer shapes.
- **`tests/test_crud_operations.py`**: Tests Create, Read, Filter, Update, Delete, and the full demo runner on an isolated mock database.
- **`tests/test_evaluation.py`**: Tests $R^2$, MAE, RMSE with synthetic matrices and model comparison sorting logic.

Execute the test suite:
```powershell
pytest -v
```
*Result: 21 passed in ~2.4 seconds.*

---

## 15. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **`ServerSelectionTimeoutError`** | MongoDB service is not started on Windows | Run PowerShell as Admin: `net start MongoDB` or verify port 27017 in `.env`. |
| **`UnicodeEncodeError` in terminal** | Windows console code page (cp1252) | Handled automatically in `main.py` via UTF-8 reconfiguration and ASCII fallback tokens. |
| **`FileNotFoundError: Trained model artifact not found`** | Model pipeline has not been trained yet | Run `python main.py train` to generate `.joblib` files in `models/`. |
| **Streamlit `Command not found`** | Streamlit script path not in system PATH | Run via Python module: `python -m streamlit run app.py` |
| **Permission error creating virtual environment** | Windows execution policy restriction | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |

---

## 16. Project Submission Packaging

The project is packaged for academic evaluation as a clean archive excluding virtual environments, secrets, and caches:

```powershell
Compress-Archive -Path data, src, models, outputs, tests, app.py, main.py, requirements.txt, README.md, .env.example, .gitignore -DestinationPath Medical_Insurance_MongoDB_CA2.zip -Force
```

**Submitted Archive:** `Medical_Insurance_MongoDB_CA2.zip`
