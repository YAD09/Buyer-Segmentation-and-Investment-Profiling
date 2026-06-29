from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.segmentation import format_currency, load_raw_data, run_segmentation


def write_research_report(result, output_path: Path) -> None:
    data = result.data
    profiles = result.cluster_profiles.copy()
    best_row = result.model_metrics.sort_values("silhouette_score", ascending=False).head(1)
    best_k = int(best_row["k"].iloc[0]) if not best_row.empty else 4
    best_score = float(best_row["silhouette_score"].iloc[0]) if not best_row.empty else 0.0

    total_clients = len(data)
    investment_rate = data["investment_buyer"].mean()
    loan_rate = data["loan_applied_flag"].mean()
    total_investment = data["total_investment"].sum()
    avg_satisfaction = data["satisfaction_score"].mean()

    top_countries = (
        data.groupby("country")
        .agg(clients=("client_id", "count"), investment_value=("total_investment", "sum"))
        .sort_values("investment_value", ascending=False)
        .head(8)
        .reset_index()
    )

    lines = [
        "# Machine Learning Based Buyer Segmentation and Investment Profiling",
        "",
        "## Executive Summary",
        "",
        f"The analysis covers {total_clients:,} real estate buyers and {format_currency(total_investment)} in linked property transactions. "
        f"{investment_rate:.1%} of clients are classified as investment-purpose buyers, {loan_rate:.1%} applied for loans, "
        f"and the average satisfaction score is {avg_satisfaction:.2f} out of 5.",
        "",
        "## Data Preparation",
        "",
        "- Removed duplicate client records by `client_id`.",
        "- Normalized categorical labels for client type, purchase purpose, loan status, and channels.",
        "- Parsed mixed date formats and converted `date_of_birth` into age.",
        "- Converted property prices from currency strings into numeric values.",
        "- Aggregated sold property activity per client, including unit count, total investment, average sale price, and office/apartment mix.",
        "",
        "## Modeling Methodology",
        "",
        "- Numeric variables were scaled with `StandardScaler`.",
        "- Categorical variables were encoded with one-hot encoding.",
        "- K-Means clustering was used for the primary segmentation model.",
        "- Agglomerative hierarchical clustering was also run to compare segment structure.",
        f"- The highest silhouette score in the tested range was k={best_k} with a score of {best_score:.3f}. The dashboard defaults to four clusters to match the requested buyer-segment framework.",
        "",
        "## Segment Profiles",
        "",
        "| Segment | Buyers | Avg Age | Loan Rate | Investment Rate | Corporate Rate | Avg Investment | Avg Sale Price |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in profiles.iterrows():
        lines.append(
            f"| {row['segment_name']} | {int(row['buyers']):,} | {row['avg_age']:.1f} | "
            f"{row['loan_rate']:.1%} | {row['investment_rate']:.1%} | {row['corporate_rate']:.1%} | "
            f"{format_currency(row['avg_total_investment'])} | {format_currency(row['avg_sale_price'])} |"
        )

    lines.extend(
        [
            "",
            "## Geographic Buyer Intelligence",
            "",
            "| Country | Clients | Linked Investment Value |",
            "|---|---:|---:|",
        ]
    )
    for _, row in top_countries.iterrows():
        lines.append(
            f"| {row['country']} | {int(row['clients']):,} | {format_currency(row['investment_value'])} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Target global and international investors with yield-oriented messaging and cross-border onboarding support.",
            "- Route loan-dependent first-time buyers into mortgage education, affordability calculators, and pre-approval journeys.",
            "- Treat corporate buyers as account-based opportunities with bulk inventory, office/apartment mix, and relationship-led sales.",
            "- Build luxury investor campaigns around high-value inventory, service quality, and repeat-purchase potential.",
            "",
            "## Deliverables",
            "",
            "- `outputs/client_segments.csv`: buyer-level segmentation output.",
            "- `outputs/cluster_profiles.csv`: descriptive statistics for each buyer segment.",
            "- `outputs/model_metrics.csv`: elbow and silhouette scores by k.",
            "- `app.py`: Streamlit dashboard for live analytics and filtering.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run buyer segmentation analysis.")
    parser.add_argument("--clients", default="data/clients.csv", help="Path to clients.csv")
    parser.add_argument("--properties", default="data/properties.csv", help="Path to properties.csv")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters for the main model")
    parser.add_argument("--outputs", default="outputs", help="Output directory")
    args = parser.parse_args()

    clients, properties = load_raw_data(args.clients, args.properties)
    result = run_segmentation(clients, properties, n_clusters=args.clusters)

    output_dir = Path(args.outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.data.to_csv(output_dir / "client_segments.csv", index=False)
    result.cluster_profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    result.model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    write_research_report(result, Path("reports") / "research_paper.md")

    print("Analysis complete")
    print(f"Client segments: {output_dir / 'client_segments.csv'}")
    print(f"Cluster profiles: {output_dir / 'cluster_profiles.csv'}")
    print(f"Model metrics: {output_dir / 'model_metrics.csv'}")
    print("Research report: reports/research_paper.md")


if __name__ == "__main__":
    main()

