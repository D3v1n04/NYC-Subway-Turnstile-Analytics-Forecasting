from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import holidays


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date_only"] = pd.to_datetime(df["Date_only"])
    df = df.sort_values("Date_only").reset_index(drop=True)

    # Target
    df["Total_Traffic"] = df["Entry_Diff"].fillna(0) + df["Exit_Diff"].fillna(0)

    # Time index
    df["t"] = np.arange(len(df))

    # Calendar features
    df["Day_of_Week"] = df["Date_only"].dt.dayofweek
    df["Month"] = df["Date_only"].dt.month
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)

    # Cyclical weekday encoding (better than raw 0..6 sometimes)
    df["Day_Sin"] = np.sin(2 * np.pi * df["Day_of_Week"] / 7.0)
    df["Day_Cos"] = np.cos(2 * np.pi * df["Day_of_Week"] / 7.0)

    # Year seasonality (captures summer/winter patterns)
    df["Year_Sin"] = np.sin(2 * np.pi * df["t"] / 365.0)
    df["Year_Cos"] = np.cos(2 * np.pi * df["t"] / 365.0)

    # Holiday flag
    years = sorted(df["Date_only"].dt.year.unique().tolist())
    us_holidays = holidays.US(years=years)
    df["Is_Holiday"] = df["Date_only"].dt.date.apply(lambda d: 1 if d in us_holidays else 0)

    # Lags (past values)
    df["Lag_1"] = df["Total_Traffic"].shift(1)
    df["Lag_7"] = df["Total_Traffic"].shift(7)
    df["Lag_14"] = df["Total_Traffic"].shift(14)
    df["Lag_28"] = df["Total_Traffic"].shift(28)

    # Rolling (PAST-only, no leakage)
    df["Rolling_7"] = df["Total_Traffic"].shift(1).rolling(7).mean()
    df["Rolling_14"] = df["Total_Traffic"].shift(1).rolling(14).mean()
    df["Rolling_30"] = df["Total_Traffic"].shift(1).rolling(30).mean()

    # Drop NaNs from lags/rolling
    df = df.dropna().reset_index(drop=True)
    return df


def run(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    data_path = output_dir / "daily_traffic_2022.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path.resolve()}")

    df = pd.read_csv(data_path)
    df = build_features(df)

    # Stronger feature set (matches what we used in the “A” model style)
    features = [
        "t",
        "Month",
        "Is_Weekend",
        "Is_Holiday",
        "Day_Sin",
        "Day_Cos",
        "Year_Sin",
        "Year_Cos",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Lag_28",
        "Rolling_7",
        "Rolling_14",
        "Rolling_30",
    ]

    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()

    X_train = train[features]
    y_train = train["Total_Traffic"]

    X_test = test[features]
    y_test = test["Total_Traffic"]

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

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("\nXGBoost Model Performance:")
    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test["Date_only"], y_test.values, label="Actual")
    plt.plot(test["Date_only"], preds, label="Predicted")
    plt.title("Daily Subway Traffic Forecast (XGBoost) — Test Set")
    plt.xlabel("Date")
    plt.ylabel("Total Traffic")
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "forecast_xgb_vs_actual.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nForecast plot saved to: {plot_path.resolve()}")

    # Save predictions
    pred_df = pd.DataFrame({
        "Date_only": test["Date_only"].values,
        "Actual": y_test.values,
        "Predicted": preds
    })
    pred_csv = output_dir / "forecast_xgb_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"Predictions saved to: {pred_csv.resolve()}")

    # Feature importance
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nTop Feature Importances:")
    print(imp_df.head(10).to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.bar(imp_df["Feature"], imp_df["Importance"].values)
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    imp_plot = output_dir / "forecast_xgb_feature_importance.png"
    plt.savefig(imp_plot, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to: {imp_plot.resolve()}")


if __name__ == "__main__":
    run(Path("outputs"))
