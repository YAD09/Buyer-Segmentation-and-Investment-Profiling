from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib

from src.segmentation import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_raw_data,
    run_segmentation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the buyer segmentation model.")
    parser.add_argument("--clients", default="data/clients.csv", help="Path to clients.csv")
    parser.add_argument("--properties", default="data/properties.csv", help="Path to properties.csv")
    parser.add_argument("--clusters", type=int, default=4, help="Number of K-Means clusters")
    parser.add_argument("--model-dir", default="models", help="Directory for trained model artifacts")
    parser.add_argument("--outputs", default="outputs", help="Directory for generated CSV outputs")
    args = parser.parse_args()

    clients, properties = load_raw_data(args.clients, args.properties)
    result = run_segmentation(clients, properties, n_clusters=args.clusters)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.outputs)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_map = (
        result.cluster_profiles[["cluster", "segment_name"]]
        .set_index("cluster")["segment_name"]
        .to_dict()
    )
    best_metric = result.model_metrics.sort_values("silhouette_score", ascending=False).iloc[0]

    artifact = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_clusters": args.clusters,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "preprocessing_pipeline": result.preprocessing_pipeline,
        "kmeans_model": result.kmeans_model,
        "segment_name_map": segment_map,
        "cluster_profiles": result.cluster_profiles,
        "model_metrics": result.model_metrics,
        "training_rows": len(result.data),
        "best_silhouette_k": int(best_metric["k"]),
        "best_silhouette_score": float(best_metric["silhouette_score"]),
    }

    model_path = model_dir / "buyer_segmentation_model.joblib"
    joblib.dump(artifact, model_path)

    result.data.to_csv(output_dir / "client_segments.csv", index=False)
    result.cluster_profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    result.model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)

    print("Model training complete")
    print(f"Training rows: {len(result.data):,}")
    print(f"Saved model: {model_path}")
    print(f"Best silhouette k: {int(best_metric['k'])}")
    print(f"Best silhouette score: {float(best_metric['silhouette_score']):.3f}")


if __name__ == "__main__":
    main()

