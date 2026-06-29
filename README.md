# Buyer Segmentation and Investment Profiling

Machine learning project for AI-based buyer segmentation and investment profiling in a real estate market intelligence setting.

## Project Contents

- `app.py` - Streamlit dashboard with segmentation overview, investor behavior, geographic analysis, and segment insights.
- `train_model.py` - trains and saves the clustering model from the supplied CSV files.
- `run_analysis.py` - command-line analysis pipeline that exports CSV outputs and a research report.
- `src/segmentation.py` - reusable data cleaning, feature engineering, K-Means, and hierarchical clustering logic.
- `data/clients.csv` - buyer dataset copied from the provided source file.
- `data/properties.csv` - property dataset copied from the provided source file.
- `reports/research_paper.md` - generated project report after running the pipeline.
- `outputs/` - generated model outputs after running the pipeline.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run the Analysis

```powershell
python run_analysis.py
```

This creates:

- `outputs/client_segments.csv`
- `outputs/cluster_profiles.csv`
- `outputs/model_metrics.csv`
- `reports/research_paper.md`

## Train the Model

```powershell
python train_model.py
```

This trains the preprocessing pipeline and K-Means clustering model on `data/clients.csv` and `data/properties.csv`, then saves:

- `models/buyer_segmentation_model.joblib`
- `outputs/client_segments.csv`
- `outputs/cluster_profiles.csv`
- `outputs/model_metrics.csv`

## Run the Dashboard

```powershell
streamlit run app.py
```

The dashboard supports filters for country, region, acquisition purpose, and client type. The default model uses four K-Means clusters to match the project requirement, and the overview tab also shows elbow and silhouette diagnostics.

## Methodology

1. Clean buyer attributes and normalize labels.
2. Parse date of birth and calculate buyer age.
3. Convert sale prices into numeric values.
4. Aggregate property behavior per client.
5. Encode categorical fields with one-hot encoding.
6. Scale numeric features with `StandardScaler`.
7. Fit K-Means clustering and compare k values with inertia and silhouette score.
8. Fit hierarchical clustering as a secondary validation model.
9. Assign descriptive business segment names using cluster profiles.
