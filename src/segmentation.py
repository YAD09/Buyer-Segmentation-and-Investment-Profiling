from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age",
    "satisfaction_score",
    "property_count",
    "total_investment",
    "avg_sale_price",
    "avg_floor_area_sqft",
    "office_ratio",
]

CATEGORICAL_FEATURES = [
    "client_type",
    "gender",
    "country",
    "region",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
]


@dataclass
class SegmentationResult:
    data: pd.DataFrame
    cluster_profiles: pd.DataFrame
    model_metrics: pd.DataFrame
    features: pd.DataFrame
    transformed_features: np.ndarray
    kmeans_model: KMeans
    preprocessing_pipeline: ColumnTransformer
    hierarchical_labels: np.ndarray


def load_raw_data(
    clients_path: str | Path = "data/clients.csv",
    properties_path: str | Path = "data/properties.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(clients_path)
    properties = pd.read_csv(properties_path)
    return clients, properties


def parse_mixed_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    dayfirst = "-" in text
    return pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)


def clean_clients(clients: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = clients.copy()
    df.columns = [col.strip() for col in df.columns]

    text_columns = [
        "client_id",
        "client_type",
        "first_name",
        "last_name",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    label_columns = [
        "client_type",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel",
    ]
    for column in label_columns:
        if column in df.columns:
            df[column] = df[column].replace({"": np.nan}).fillna("Unknown")

    df["client_type"] = df["client_type"].replace(
        {"Corporate": "Company", "corporate": "Company", "company": "Company"}
    )
    df["acquisition_purpose"] = df["acquisition_purpose"].replace(
        {"Personal use": "Home", "Personal Use": "Home", "personal use": "Home"}
    )
    df["loan_applied"] = df["loan_applied"].replace(
        {"Y": "Yes", "N": "No", "yes": "Yes", "no": "No", "TRUE": "Yes", "FALSE": "No"}
    )

    df["date_of_birth_parsed"] = df["date_of_birth"].apply(parse_mixed_date)
    as_of_date = as_of_date or pd.Timestamp.today().normalize()
    df["age"] = ((as_of_date - df["date_of_birth_parsed"]).dt.days / 365.25).round(1)
    df["age"] = df["age"].clip(lower=18, upper=100)
    df["age"] = df["age"].fillna(df["age"].median())

    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())

    return df.drop_duplicates(subset=["client_id"], keep="first")


def clean_properties(properties: pd.DataFrame) -> pd.DataFrame:
    df = properties.copy()
    df.columns = [col.strip() for col in df.columns]

    df["client_ref"] = df["client_ref"].replace({"": np.nan})
    df["sale_price_num"] = (
        df["sale_price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )
    df["floor_area_sqft"] = pd.to_numeric(df["floor_area_sqft"], errors="coerce")
    df["transaction_date_parsed"] = df["transaction_date"].apply(parse_mixed_date)

    return df


def build_property_features(properties: pd.DataFrame) -> pd.DataFrame:
    sold = properties[properties["client_ref"].notna()].copy()
    if sold.empty:
        return pd.DataFrame(columns=["client_ref"])

    grouped = sold.groupby("client_ref")
    features = grouped.agg(
        property_count=("listing_id", "count"),
        total_investment=("sale_price_num", "sum"),
        avg_sale_price=("sale_price_num", "mean"),
        max_sale_price=("sale_price_num", "max"),
        avg_floor_area_sqft=("floor_area_sqft", "mean"),
        office_units=("unit_category", lambda values: (values == "Office").sum()),
        apartment_units=("unit_category", lambda values: (values == "Apartment").sum()),
        first_purchase_date=("transaction_date_parsed", "min"),
        latest_purchase_date=("transaction_date_parsed", "max"),
    ).reset_index()

    features["office_ratio"] = features["office_units"] / features["property_count"].replace(0, np.nan)
    features["apartment_ratio"] = features["apartment_units"] / features["property_count"].replace(0, np.nan)
    features[["office_ratio", "apartment_ratio"]] = features[
        ["office_ratio", "apartment_ratio"]
    ].fillna(0)

    return features


def build_modeling_dataset(clients: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    clean_client_data = clean_clients(clients)
    clean_property_data = clean_properties(properties)
    property_features = build_property_features(clean_property_data)

    df = clean_client_data.merge(property_features, left_on="client_id", right_on="client_ref", how="left")
    fill_zero = [
        "property_count",
        "total_investment",
        "avg_sale_price",
        "max_sale_price",
        "avg_floor_area_sqft",
        "office_units",
        "apartment_units",
        "office_ratio",
        "apartment_ratio",
    ]
    for column in fill_zero:
        df[column] = df[column].fillna(0)

    df["investment_buyer"] = (df["acquisition_purpose"] == "Investment").astype(int)
    df["loan_applied_flag"] = (df["loan_applied"] == "Yes").astype(int)
    df["corporate_flag"] = (df["client_type"] == "Company").astype(int)
    df["international_flag"] = (df["country"] != "USA").astype(int)
    df["luxury_flag"] = (
        df["avg_sale_price"] >= df["avg_sale_price"].quantile(0.75)
    ).astype(int)

    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def evaluate_k_range(transformed_features: np.ndarray, k_values: Iterable[int]) -> pd.DataFrame:
    rows = []
    n_samples = transformed_features.shape[0]
    for k in k_values:
        if k < 2 or k >= n_samples:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(transformed_features)
        inertia = float(model.inertia_)
        silhouette = float(silhouette_score(transformed_features, labels))
        rows.append({"k": k, "inertia": inertia, "silhouette_score": silhouette})
    return pd.DataFrame(rows)


def run_segmentation(
    clients: pd.DataFrame,
    properties: pd.DataFrame,
    n_clusters: int = 4,
    k_min: int = 2,
    k_max: int = 8,
) -> SegmentationResult:
    dataset = build_modeling_dataset(clients, properties)
    features = get_model_features(dataset)
    preprocessor = make_preprocessor()
    transformed = preprocessor.fit_transform(features)

    metrics = evaluate_k_range(transformed, range(k_min, k_max + 1))

    kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=RANDOM_STATE)
    dataset["cluster"] = kmeans.fit_predict(transformed)

    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(transformed)
    dataset["hierarchical_cluster"] = hierarchical_labels

    profiles = build_cluster_profiles(dataset)
    name_map = assign_segment_names(profiles)
    dataset["segment_name"] = dataset["cluster"].map(name_map)
    profiles["segment_name"] = profiles["cluster"].map(name_map)
    profiles = profiles.sort_values("segment_name").reset_index(drop=True)

    return SegmentationResult(
        data=dataset,
        cluster_profiles=profiles,
        model_metrics=metrics,
        features=features,
        transformed_features=transformed,
        kmeans_model=kmeans,
        preprocessing_pipeline=preprocessor,
        hierarchical_labels=hierarchical_labels,
    )


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        df.groupby("cluster")
        .agg(
            buyers=("client_id", "count"),
            avg_age=("age", "mean"),
            median_age=("age", "median"),
            avg_satisfaction=("satisfaction_score", "mean"),
            loan_rate=("loan_applied_flag", "mean"),
            investment_rate=("investment_buyer", "mean"),
            corporate_rate=("corporate_flag", "mean"),
            international_rate=("international_flag", "mean"),
            luxury_rate=("luxury_flag", "mean"),
            avg_property_count=("property_count", "mean"),
            avg_total_investment=("total_investment", "mean"),
            avg_sale_price=("avg_sale_price", "mean"),
            office_ratio=("office_ratio", "mean"),
        )
        .reset_index()
    )

    for column in [
        "loan_rate",
        "investment_rate",
        "corporate_rate",
        "international_rate",
        "luxury_rate",
        "office_ratio",
    ]:
        profiles[column] = profiles[column].fillna(0)

    return profiles


def assign_segment_names(profiles: pd.DataFrame) -> dict[int, str]:
    profile_data = profiles.set_index("cluster").copy()
    assigned: dict[int, str] = {}
    remaining = set(profile_data.index.tolist())

    ordered_rules = [
        (
            "C3 - Corporate Buyers",
            lambda data: data["corporate_rate"] * 2
            + data["office_ratio"]
            + data["avg_property_count"] / max(data["avg_property_count"].max(), 1),
        ),
        (
            "C4 - Luxury Investors",
            lambda data: data["luxury_rate"] * 2
            + data["avg_sale_price"] / max(data["avg_sale_price"].max(), 1)
            + data["avg_satisfaction"] / 5,
        ),
        (
            "C1 - Global Investors",
            lambda data: data["investment_rate"] * 2
            + data["international_rate"]
            + data["avg_total_investment"] / max(data["avg_total_investment"].max(), 1),
        ),
        (
            "C2 - First-Time Buyers",
            lambda data: data["loan_rate"] * 2
            - data["median_age"] / 100
            + (1 - data["avg_property_count"] / max(data["avg_property_count"].max(), 1)),
        ),
    ]

    for label, scorer in ordered_rules:
        if not remaining:
            break
        scores = scorer(profile_data.loc[list(remaining)])
        cluster = int(scores.idxmax())
        assigned[cluster] = label
        remaining.remove(cluster)

    for index, cluster in enumerate(sorted(remaining), start=1):
        assigned[int(cluster)] = f"C{index + 4} - Emerging Segment"

    return assigned


def filter_segments(
    df: pd.DataFrame,
    countries: list[str] | None = None,
    regions: list[str] | None = None,
    purposes: list[str] | None = None,
    client_types: list[str] | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    if purposes:
        filtered = filtered[filtered["acquisition_purpose"].isin(purposes)]
    if client_types:
        filtered = filtered[filtered["client_type"].isin(client_types)]
    return filtered


def format_currency(value: float) -> str:
    return f"${value:,.0f}"
