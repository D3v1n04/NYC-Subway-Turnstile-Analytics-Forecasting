# NYC Subway Turnstile Analytics & Forecasting (2022)

## Overview

This project analyzes and models NYC subway turnstile data from 2022.

The goal was to take a large real-world dataset (~10.9 million records), properly clean and process it, extract meaningful traffic patterns, and build forecasting models to predict daily subway ridership.

This project combines:

- Data analytics
- Time-series feature engineering
- Machine learning using XGBoost


## Project Objectives

- Clean and process raw MTA turnstile logs
- Perform exploratory data analysis (EDA)
- Identify traffic patterns across:
  - Days of week
  - Weekdays vs weekends
  - Hours of day
  - Monthly totals
  - Station-level activity
- Build forecasting models to predict daily traffic
- Evaluate multi-horizon forecasting performance


## Dataset

**Source:** NYC MTA Turnstile Usage Data (2022)  
**Size:** ~10.9 million interval-level records  

Each record includes:

- Station / Unit
- Date
- Time
- Entries (cumulative)
- Exits (cumulative)

IMPORTANT NOTE: The raw dataset is not included in this repository due to file size constraints.


## Data Cleaning & Processing

Raw MTA turnstile data consists of cumulative counters rather than direct traffic counts.

Key steps:

- Combined `Date` and `Time` into a proper datetime column
- Sorted records per turnstile before computing traffic deltas
- Computed interval traffic using differences (`Entry_Diff`, `Exit_Diff`)
- Removed negative counter resets and unrealistic spikes
- Aggregated interval-level data into daily totals for modeling

Final cleaned dataset:

- 365 daily traffic observations (2022)


## Exploratory Data Analysis (EDA)

Generated insights include:

- Daily total traffic trend
- Weekday vs Weekend comparison (~66% higher weekday traffic)
- Weekly seasonality breakdown
- Monthly recovery curve
- Top 10 busiest stations
- Hour-by-day traffic heatmap
- Station-level hourly patterns

All visualizations and CSV outputs are saved in the `/outputs` directory.


## Forecasting Approach

**Model:** XGBoost Regressor

### Feature Engineering

- Lag features (1, 7, 14, 21, 28 days)
- Rolling averages (7, 14, 30 days)
- Weekend indicator
- U.S. holiday indicator
- Cyclical encoding of day-of-week
- Time index feature

### Train/Test Split

- 80% training
- 20% testing
- Time-based split (no shuffling)


## Forecasting Results (Test Set)

- MAE: ~336,000 riders
- RMSE: ~510,000 riders

Average weekday traffic is ~5.3 million riders, corresponding to roughly a 6–7% prediction error.


## Multi-Horizon Forecasting

Direct models built for:

- 1-day ahead
- 7-day ahead
- 14-day ahead
- 30-day ahead

Performance degrades gradually as forecast horizon increases, which is expected behavior in real-world time-series forecasting.


## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost


## Project Structure

 - data/        # Raw dataset (not included)
 - src/         # Data loading, cleaning, analysis, modeling
 - outputs/     # Generated plots and prediction outputs


## How to Run
1. Create a virtual environment
   - python -m venv .venv
   - source .venv/bin/activate
   - pip install -r requirements.txt

2. Add dataset
   - Place the raw file inside: data/MTA_Subway_Turnstile_Usage_Data_2022.csv

3. Run the full pipeline
   - python src/main.py
