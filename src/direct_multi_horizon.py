from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import holidays


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def run(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load daily data (already aggregated)
    data_path = output_dir / "daily_traffic_2022.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path.resolve()}")

    df = pd.read_csv(data_path)
    df["Date_only"] = pd.to_datetime(df["Date_only"])
    df = df.sort_values("Date_only").reset_index(drop=True)

    # Total traffic
    df["Total_Traffic"] = df["Entry_Diff"].fillna(0) + df["Exit_Diff"].fillna(0)

    # ---------------------------
    # Feature engineering (NO leakage)

    # Time index
    df["t"] = np.arange(len(df))

    # Calendar
    df["Day_of_Week"] = df["Date_only"].dt.dayofweek
    df["Day_Sin"] = np.sin(2 * np.pi * df["Day_of_Week"] / 7.0)
    df["Day_Cos"] = np.cos(2 * np.pi * df["Day_of_Week"] / 7.0)
    df["Month"] = df["Date_only"].dt.month
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)

    # Holiday flag
    us_holidays = holidays.US(years=sorted(df["Date_only"].dt.year.unique().tolist()))
    df["Is_Holiday"] = df["Date_only"].dt.date.apply(lambda d: 1 if d in us_holidays else 0)

    # Yearly seasonality (365-day cycle)
    df["Year_Sin"] = np.sin(2 * np.pi * df["t"] / 365.0)
    df["Year_Cos"] = np.cos(2 * np.pi * df["t"] / 365.0)

    # Lags (past values only)
    df["Lag_1"]  = df["Total_Traffic"].shift(1)
    df["Lag_7"]  = df["Total_Traffic"].shift(7)
    df["Lag_14"] = df["Total_Traffic"].shift(14)
    df["Lag_21"] = df["Total_Traffic"].shift(21)
    df["Lag_28"] = df["Total_Traffic"].shift(28)

    # Rolling means using PAST only (shift first!)
    df["Rolling_7"]  = df["Total_Traffic"].shift(1).rolling(7).mean()
    df["Rolling_14"] = df["Total_Traffic"].shift(1).rolling(14).mean()
    df["Rolling_30"] = df["Total_Traffic"].shift(1).rolling(30).mean()

    # Base feature list used for all horizons
    FEATURES = [
        "t", "Month",
        "Is_Weekend", "Is_Holiday",
        "Day_Sin", "Day_Cos",
        "Year_Sin", "Year_Cos",
        "Lag_1", "Lag_7", "Lag_14", "Lag_21", "Lag_28",
        "Rolling_7", "Rolling_14", "Rolling_30",
    ]

    # Drop rows that can't have features (because of lag/rolling)
    df = df.dropna().reset_index(drop=True)

    # ---------------------------
    # Train/Test split by time
    train_size = int(len(df) * 0.8)
    train_df_base = df.iloc[:train_size].copy()
    test_df_base  = df.iloc[train_size:].copy()

    # ---------------------------
    # Train direct models per horizon
    HORIZONS = [1, 7, 14, 30]
    results = []

    for H in HORIZONS:
        train_df = train_df_base.copy()
        test_df  = test_df_base.copy()

        # Create horizon target: y(t+H)
        target_col = f"y_{H}"
        train_df[target_col] = train_df["Total_Traffic"].shift(-H)
        test_df[target_col]  = test_df["Total_Traffic"].shift(-H)

        # Drop rows where target doesn't exist (near the end)
        train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
        test_df  = test_df.dropna(subset=[target_col]).reset_index(drop=True)

        X_train = train_df[FEATURES]
        y_train = train_df[target_col]

        X_test = test_df[FEATURES]
        y_test = test_df[target_col]

        # XGBoost model (stable defaults)
        model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        H_mae = _mae(y_test, preds)
        H_rmse = _rmse(y_test, preds)

        results.append({"H": H, "MAE": H_mae, "RMSE": H_rmse})

        # Save predictions
        pred_out = pd.DataFrame({
            "Date": test_df["Date_only"],
            "Actual": y_test.values,
            "Predicted": preds
        })
        pred_csv = output_dir / f"direct_h{H}_predictions.csv"
        pred_out.to_csv(pred_csv, index=False)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(test_df["Date_only"], y_test.values, label="Actual")
        plt.plot(test_df["Date_only"], preds, label=f"Predicted (H={H})")
        plt.title(f"Direct Multi-Horizon Forecast — H={H} days (Test Set)")
        plt.xlabel("Date")
        plt.ylabel("Total Traffic")
        plt.legend()
        plt.tight_layout()

        plot_path = output_dir / f"direct_h{H}_forecast.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"\nH={H} days -> MAE: {H_mae:,.0f} | RMSE: {H_rmse:,.0f}")
        print(f"Saved: {pred_csv.resolve()}")
        print(f"Saved: {plot_path.resolve()}")

    # Summary table
    results_df = pd.DataFrame(results).sort_values("H").reset_index(drop=True)
    summary_csv = output_dir / "direct_multi_horizon_summary.csv"
    results_df.to_csv(summary_csv, index=False)

    print("\nDIRECT MULTI-HORIZON SUMMARY:")
    print(results_df.to_string(index=False))
    print(f"\nSaved summary: {summary_csv.resolve()}")


if __name__ == "__main__":
    run(Path("outputs"))
