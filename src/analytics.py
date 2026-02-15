import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def savefig(path: Path, dpi: int = 300):
    plt.savefig(path, dpi=dpi)
    plt.close()


def run(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    # Create shared columns
    df = df.copy()
    df["Date_only"] = df["Datetime"].dt.floor("D")
    df["Hour"] = df["Datetime"].dt.hour
    df["Weekday_Name"] = df["Datetime"].dt.day_name()
    df["Interval_Traffic"] = df["Entry_Diff"].fillna(0) + df["Exit_Diff"].fillna(0)

# ----------------------------------------------------------------------------------
    # Daily totals
    daily_traffic = df.groupby("Date_only")[["Entry_Diff", "Exit_Diff"]].sum()
    daily_traffic["Total_Traffic"] = daily_traffic["Entry_Diff"] + daily_traffic["Exit_Diff"]

    print("\nDaily traffic (first 5 days):")
    print(daily_traffic.head())

    daily_traffic["Total_Traffic"].plot(figsize=(12, 6))
    plt.title("NYC Subway Total Daily Traffic (2022)")
    plt.xlabel("Date")
    plt.ylabel("Turnstile Traffic (Entries + Exits)")
    plt.tight_layout()

    daily_plot = output_dir / "daily_traffic_2022.png"
    savefig(daily_plot)
    print(f"\nPlot saved to: {daily_plot.resolve()}")

    daily_csv = output_dir / "daily_traffic_2022.csv"
    daily_traffic.to_csv(daily_csv)
    print(f"Daily totals saved to: {daily_csv.resolve()}")

    # ----------------------------------------------------------------------------------
    # Busiest stations
    print("\nBusiest stations")

    station_traffic = df.groupby("Unit")[["Entry_Diff", "Exit_Diff"]].sum()
    station_traffic["Total_Traffic"] = station_traffic["Entry_Diff"] + station_traffic["Exit_Diff"]
    station_traffic = station_traffic.sort_values("Total_Traffic", ascending=False)

    print("\nTop 10 Busiest Stations (Units):")
    print(station_traffic.head(10))

    station_csv = output_dir / "station_traffic_2022.csv"
    station_traffic.to_csv(station_csv)
    print(f"\nFull station ranking saved to: {station_csv.resolve()}")

    top10 = station_traffic.head(10)
    plt.figure(figsize=(10, 6))
    top10["Total_Traffic"].plot(kind="bar")
    plt.title("Top 10 Busiest Subway Stations (2022)")
    plt.ylabel("Total Turnstile Traffic")
    plt.xlabel("Station (Unit)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    top10_plot = output_dir / "top10_stations_2022.png"
    savefig(top10_plot)
    print(f"Top 10 plot saved to: {top10_plot.resolve()}")

    print("\nMetadata for Top 10 Units:")
    for unit in top10.index:
        example = df[df["Unit"] == unit].iloc[0]
        print(f"Unit: {unit} | C/A: {example['C/A']} | SCP: {example['SCP']}")

# ----------------------------------------------------------------------------------
    # Hourly avg traffic for Top 5 stations
    print("\nStation hourly breakdown")

    top5_units = top10.index[:5].tolist()
    print("Top 5 Units:", top5_units)

    top5_df = df[df["Unit"].isin(top5_units)].copy()

    station_hourly = top5_df.groupby(["Unit", "Hour"])[["Entry_Diff", "Exit_Diff"]].mean()
    station_hourly["Avg_Traffic"] = station_hourly["Entry_Diff"] + station_hourly["Exit_Diff"]
    station_hourly = station_hourly.reset_index()

    print("\nStation hourly data (sample):")
    print(station_hourly.head())

    plt.figure(figsize=(12, 6))
    for unit in top5_units:
        station_data = station_hourly[station_hourly["Unit"] == unit]
        plt.plot(station_data["Hour"], station_data["Avg_Traffic"], marker="o", label=unit)

    plt.title("Hourly Average Traffic – Top 5 Stations (2022)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Interval Traffic")
    plt.legend()
    plt.xticks(range(0, 24))
    plt.tight_layout()

    top5_plot = output_dir / "top5_station_hourly_patterns_2022.png"
    savefig(top5_plot)
    print(f"Station-level hourly plot saved to: {top5_plot.resolve()}")

# ----------------------------------------------------------------------------------
    # Monthly recovery curve
    print("\nMonthly recovery curve")

    df["Month"] = df["Datetime"].dt.to_period("M")
    monthly_traffic = df.groupby("Month")[["Entry_Diff", "Exit_Diff"]].sum()
    monthly_traffic["Total_Traffic"] = monthly_traffic["Entry_Diff"] + monthly_traffic["Exit_Diff"]
    monthly_traffic.index = monthly_traffic.index.to_timestamp()

    print("\nMonthly traffic totals:")
    print(monthly_traffic)

    monthly_csv = output_dir / "monthly_traffic_2022.csv"
    monthly_traffic.to_csv(monthly_csv)
    print(f"\nMonthly totals saved to: {monthly_csv.resolve()}")

    monthly_traffic["Total_Traffic"].plot(marker="o", figsize=(10, 6))
    plt.title("NYC Subway Monthly Total Traffic (2022)")
    plt.xlabel("Month")
    plt.ylabel("Total Turnstile Traffic")
    plt.tight_layout()

    monthly_plot = output_dir / "monthly_traffic_2022.png"
    savefig(monthly_plot)
    print(f"Monthly plot saved to: {monthly_plot.resolve()}")

    # ----------------------------------------------------------------------------------
    # Weekday vs Weekend split
    print("\nWeekday vs Weekend split")

    df["Day_of_Week"] = df["Datetime"].dt.dayofweek
    df["Day_Type"] = df["Day_of_Week"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

    daily_by_type = df.groupby(["Date_only", "Day_Type"])[["Entry_Diff", "Exit_Diff"]].sum()
    daily_by_type["Total_Traffic"] = daily_by_type["Entry_Diff"] + daily_by_type["Exit_Diff"]

    daytype_avg = daily_by_type.groupby("Day_Type")["Total_Traffic"].mean()
    print("\nAverage Daily Traffic:")
    print(daytype_avg)

    percent_diff = ((daytype_avg["Weekday"] - daytype_avg["Weekend"]) / daytype_avg["Weekend"]) * 100
    print(f"\nWeekdays have {percent_diff:.2f}% more traffic than weekends.")

    weekday_csv = output_dir / "weekday_weekend_traffic_2022.csv"
    daytype_avg.to_csv(weekday_csv)
    print(f"Weekday vs weekend data saved to: {weekday_csv.resolve()}")

    plt.figure(figsize=(6, 6))
    daytype_avg.plot(kind="bar")
    plt.title("Average Daily Subway Traffic: Weekday vs Weekend (2022)")
    plt.ylabel("Average Daily Traffic")
    plt.tight_layout()

    weekday_plot = output_dir / "weekday_weekend_traffic_2022.png"
    savefig(weekday_plot)
    print(f"Weekday vs weekend plot saved to: {weekday_plot.resolve()}")

# ----------------------------------------------------------------------------------
    # Weekly seasonality curve
    print("\nWeekly seasonality curve")

    daily_totals = df.groupby("Date_only")[["Entry_Diff", "Exit_Diff"]].sum()
    daily_totals["Total_Traffic"] = daily_totals["Entry_Diff"] + daily_totals["Exit_Diff"]
    daily_totals["Weekday"] = daily_totals.index.day_name()

    weekly_pattern = daily_totals.groupby("Weekday")["Total_Traffic"].mean()
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_pattern = weekly_pattern.reindex(ordered_days)

    print("\nAverage Traffic by Day of Week:")
    print(weekly_pattern)

    weekly_csv = output_dir / "weekly_pattern_2022.csv"
    weekly_pattern.to_csv(weekly_csv)
    print(f"Weekly pattern saved to: {weekly_csv.resolve()}")

    plt.figure(figsize=(10, 6))
    weekly_pattern.plot(kind="bar")
    plt.title("Average Daily Subway Traffic by Day of Week (2022)")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Daily Traffic")
    plt.tight_layout()

    weekly_plot = output_dir / "weekly_pattern_2022.png"
    savefig(weekly_plot)
    print(f"Weekly pattern plot saved to: {weekly_plot.resolve()}")

# ----------------------------------------------------------------------------------
    # Hour x Day heatmap
    print("\nBuilding Hour x Day-of-Week heatmap")

    clipped = df["Interval_Traffic"].clip(upper=5000)

    heatmap_df = (
        df.assign(Interval_Traffic_Clipped=clipped)
        .groupby(["Weekday_Name", "Hour"])["Interval_Traffic_Clipped"]
        .mean()
        .unstack("Hour")
    )

    heatmap_df = heatmap_df.reindex(ordered_days)

    heatmap_csv = output_dir / "heatmap_hour_x_weekday_2022.csv"
    heatmap_df.to_csv(heatmap_csv)
    print(f"Heatmap data saved to: {heatmap_csv.resolve()}")

    plt.figure(figsize=(14, 5))
    plt.imshow(heatmap_df.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Avg Interval Traffic (clipped)")
    plt.title("NYC Subway Avg Turnstile Traffic by Hour & Day (2022)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.xticks(ticks=range(0, 24), labels=[str(h) for h in range(24)])
    plt.yticks(ticks=range(len(heatmap_df.index)), labels=heatmap_df.index)
    plt.tight_layout()

    heatmap_plot = output_dir / "heatmap_hour_x_weekday_2022.png"
    savefig(heatmap_plot)
    print(f"Heatmap plot saved to: {heatmap_plot.resolve()}")

# ----------------------------------------------------------------------------------
    # Peak hour distribution
    print("\nPeak hour distribution")

    hourly_traffic = df.groupby("Hour")[["Entry_Diff", "Exit_Diff"]].sum()
    hourly_traffic["Total_Traffic"] = hourly_traffic["Entry_Diff"] + hourly_traffic["Exit_Diff"]

    hourly_csv = output_dir / "hourly_traffic_2022.csv"
    hourly_traffic.to_csv(hourly_csv)
    print(f"\nHourly totals saved to: {hourly_csv.resolve()}")

    plt.figure(figsize=(10, 6))
    plt.bar(hourly_traffic.index.astype(int), hourly_traffic["Total_Traffic"].values)
    plt.title("NYC Subway Total Traffic by Hour (2022)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Total Turnstile Traffic")
    plt.xticks(range(0, 24))
    plt.tight_layout()

    hourly_plot = output_dir / "hourly_traffic_2022.png"
    savefig(hourly_plot)
    print(f"Hourly plot saved to: {hourly_plot.resolve()}")


if __name__ == "__main__":
    from load_data import load_data
    df = load_data(Path("data") / "MTA_Subway_Turnstile_Usage_Data_2022.csv")
    run(df, Path("outputs"))
