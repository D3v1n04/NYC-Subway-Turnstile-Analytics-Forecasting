from pathlib import Path

from load_data import load_data
import analytics
import forecasting_xgb
import direct_multi_horizon


def main():
    data_dir = Path("data")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "MTA_Subway_Turnstile_Usage_Data_2022.csv"
    file_path = data_dir / csv_name

    print("\n=== NYC Subway Turnstile Analytics Pipeline ===\n")

    # 1) Load + clean interval-level data
    df = load_data(file_path)

    # 2) Analytics outputs (daily csv + plots)
    analytics.run(df, outputs_dir)

    # 3) Forecasting (XGB baseline)
    forecasting_xgb.run(outputs_dir)

    # 4) Direct multi-horizon models + summary table
    direct_multi_horizon.run(outputs_dir)

    print("\nPipeline complete ✅  Check outputs/.\n")


if __name__ == "__main__":
    main()
