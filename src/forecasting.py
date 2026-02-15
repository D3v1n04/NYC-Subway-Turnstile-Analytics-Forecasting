from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import holidays


def run(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load daily data
    data_path = output_dir / "daily_traffic_2022.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}")

    df = pd.read_csv(data_path)
    df["Date_only"] = pd.to_datetime(df["Date_only"])
    df = df.sort_values("Date_only").reset_index(drop=True)

    # Target
    df["Total_Traffic"] = df["Entry_Diff"] + df["Exit_Diff"]

    # ---------------------------
    # Calendar features
    df["t"] = range(len(df))
    df["Day_of_Week"] = df["Date_only"].dt.dayofweek
    df["Day_Sin"] = np.sin(2 * np.pi * df["Day_of_Week"] / 7)
    df["Day_Cos"] = np.cos(2 * np.pi * df["Day_of_Week"] / 7)
    df["Month"] = df["Date_only"].dt.month
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)

    # Holidays
    us_holidays = holidays.US(years=df["Date_only"].dt.year.unique().tolist())
    df["Is_Holiday"] = df["Date_only"].dt.date.apply(lambda d: 1 if d in us_holidays else 0)

    print("Holiday days in dataset:", int(df["Is_Holiday"].sum()))

    # ---------------------------
    # Lag + rolling features (PAST only)
    df["Lag_1"]  = df["Total_Traffic"].shift(1)
    df["Lag_7"]  = df["Total_Traffic"].shift(7)
    df["Lag_14"] = df["Total_Traffic"].shift(14)
    df["Lag_21"] = df["Total_Traffic"].shift(21)
    df["Lag_28"] = df["Total_Traffic"].shift(28)

    df["Rolling_7"]  = df["Total_Traffic"].shift(1).rolling(7).mean()
    df["Rolling_14"] = df["Total_Traffic"].shift(1).rolling(14).mean()
    df["Rolling_30"] = df["Total_Traffic"].shift(1).rolling(30).mean()

    df = df.dropna().reset_index(drop=True)

    # ---------------------------
    # Features used (must match walk-forward x_row keys)
    features = [
        "Is_Weekend",
        "Is_Holiday",
        "Day_Sin",
        "Day_Cos",
        "Month",
        "t",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Lag_21",
        "Lag_28",
        "Rolling_7",
        "Rolling_14",
        "Rolling_30",
    ]

    # ---------------------------
    # Train/Test split
    train_size = int(len(df) * 0.8)

    train_full = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()

    X_train = train_full[features]
    y_train = train_full["Total_Traffic"]

    y_test = test["Total_Traffic"].values
    test_walk = test.reset_index(drop=True)

    # ---------------------------
    # Model
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)

    # ---------------------------
    # WALK-FORWARD forecasting (recursive)
    walk_preds = []
    history = train_full["Total_Traffic"].tolist()

    for i in range(len(test_walk)):
        lag_1  = history[-1]
        lag_7  = history[-7]  if len(history) >= 7  else history[-1]
        lag_14 = history[-14] if len(history) >= 14 else lag_7
        lag_21 = history[-21] if len(history) >= 21 else lag_14
        lag_28 = history[-28] if len(history) >= 28 else lag_21

        rolling_7  = np.mean(history[-7:])  if len(history) >= 7  else np.mean(history)
        rolling_14 = np.mean(history[-14:]) if len(history) >= 14 else rolling_7
        rolling_30 = np.mean(history[-30:]) if len(history) >= 30 else rolling_14

        row = test_walk.loc[i]

        x_row = pd.DataFrame([{
            "Is_Weekend": int(row["Is_Weekend"]),
            "Is_Holiday": int(row["Is_Holiday"]),
            "Day_Sin": float(row["Day_Sin"]),
            "Day_Cos": float(row["Day_Cos"]),
            "Month": int(row["Month"]),
            "t": int(row["t"]),
            "Lag_1": lag_1,
            "Lag_7": lag_7,
            "Lag_14": lag_14,
            "Lag_21": lag_21,
            "Lag_28": lag_28,
            "Rolling_7": rolling_7,
            "Rolling_14": rolling_14,
            "Rolling_30": rolling_30,
        }])[features]

        pred = float(model.predict(x_row)[0])
        walk_preds.append(pred)
        history.append(pred)

    walk_preds = np.array(walk_preds)

    # ---------------------------
    # Evaluation
    mae = mean_absolute_error(y_test, walk_preds)
    rmse = np.sqrt(mean_squared_error(y_test, walk_preds))

    print("\nWALK-FORWARD Model Performance:")
    print(f"MAE: {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")

    # ---------------------------
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test["Date_only"], y_test, label="Actual")
    plt.plot(test["Date_only"], walk_preds, label="Predicted (Walk-Forward)")
    plt.legend()
    plt.title("Daily Subway Traffic Forecast (Walk-Forward)")
    plt.xlabel("Date")
    plt.ylabel("Total Traffic")
    plt.tight_layout()

    forecast_plot = output_dir / "forecast_walkforward_vs_actual.png"
    plt.savefig(forecast_plot, dpi=300)
    plt.close()

    print(f"\nWalk-forward forecast plot saved to: {forecast_plot.resolve()}")

    # --------------------------------------------------
    # MULTI-HORIZON EVALUATION (walk-forward)
    actual = test_walk["Total_Traffic"].values
    preds = walk_preds

    def horizon_metrics(actual_arr, preds_arr, h):
        if h < 1:
            raise ValueError("h must be >= 1")
        y_true = actual_arr[h-1:]
        y_pred = preds_arr[:- (h-1)] if (h-1) > 0 else preds_arr
        mae_h = mean_absolute_error(y_true, y_pred)
        rmse_h = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae_h, rmse_h

    horizons = [1, 7, 14, 30]
    results = []

    print("\nMULTI-HORIZON PERFORMANCE (Walk-Forward):")
    for h in horizons:
        mae_h, rmse_h = horizon_metrics(actual, preds, h)
        results.append((h, mae_h, rmse_h))
        print(f"H={h:>2} days -> MAE: {mae_h:,.0f} | RMSE: {rmse_h:,.0f}")

    horizon_df = pd.DataFrame(results, columns=["Horizon_Days", "MAE", "RMSE"])
    horizon_csv = output_dir / "forecast_walkforward_horizon_metrics.csv"
    horizon_df.to_csv(horizon_csv, index=False)
    print(f"\nHorizon metrics saved to: {horizon_csv.resolve()}")


if __name__ == "__main__":
    run(Path("outputs"))
