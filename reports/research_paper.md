# Machine Learning Based Buyer Segmentation and Investment Profiling

## Executive Summary

The analysis covers 2,000 real estate buyers and $2,520,750,961 in linked property transactions. 30.8% of clients are classified as investment-purpose buyers, 36.8% applied for loans, and the average satisfaction score is 3.03 out of 5.

## Data Preparation

- Removed duplicate client records by `client_id`.
- Normalized categorical labels for client type, purchase purpose, loan status, and channels.
- Parsed mixed date formats and converted `date_of_birth` into age.
- Converted property prices from currency strings into numeric values.
- Aggregated sold property activity per client, including unit count, total investment, average sale price, and office/apartment mix.

## Modeling Methodology

- Numeric variables were scaled with `StandardScaler`.
- Categorical variables were encoded with one-hot encoding.
- K-Means clustering was used for the primary segmentation model.
- Agglomerative hierarchical clustering was also run to compare segment structure.
- The highest silhouette score in the tested range was k=3 with a score of 0.151. The dashboard defaults to four clusters to match the requested buyer-segment framework.

## Segment Profiles

| Segment | Buyers | Avg Age | Loan Rate | Investment Rate | Corporate Rate | Avg Investment | Avg Sale Price |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 - Global Investors | 555 | 57.4 | 38.6% | 32.1% | 4.0% | $1,109,261 | $309,289 |
| C2 - First-Time Buyers | 681 | 55.3 | 35.1% | 30.5% | 5.1% | $1,059,964 | $303,451 |
| C3 - Corporate Buyers | 90 | 65.1 | 35.6% | 28.9% | 4.4% | $2,125,594 | $340,930 |
| C4 - Luxury Investors | 674 | 54.5 | 37.2% | 30.1% | 6.2% | $1,471,769 | $423,132 |

## Geographic Buyer Intelligence

| Country | Clients | Linked Investment Value |
|---|---:|---:|
| USA | 1,538 | $1,953,032,451 |
| UK | 95 | $122,026,575 |
| Canada | 85 | $102,630,946 |
| Germany | 56 | $66,871,571 |
| France | 53 | $64,899,004 |
| Belgium | 43 | $54,225,932 |
| Mexico | 40 | $48,429,747 |
| Russia | 36 | $44,661,828 |

## Recommendations

- Target global and international investors with yield-oriented messaging and cross-border onboarding support.
- Route loan-dependent first-time buyers into mortgage education, affordability calculators, and pre-approval journeys.
- Treat corporate buyers as account-based opportunities with bulk inventory, office/apartment mix, and relationship-led sales.
- Build luxury investor campaigns around high-value inventory, service quality, and repeat-purchase potential.

## Deliverables

- `outputs/client_segments.csv`: buyer-level segmentation output.
- `outputs/cluster_profiles.csv`: descriptive statistics for each buyer segment.
- `outputs/model_metrics.csv`: elbow and silhouette scores by k.
- `app.py`: Streamlit dashboard for live analytics and filtering.