from pathlib import Path
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """
    Load and clean raw MTA turnstile interval-level CSV.
    Returns cleaned dataframe with Entry_Diff and Exit_Diff.
    """

    print("Loading data from:", path.resolve())

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # Combine Date and Time into datetime
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(subset=["Datetime"])

    # Convert numeric
    df["Entries"] = pd.to_numeric(df["Entries"], errors="coerce")
    df["Exits"] = pd.to_numeric(df["Exits"], errors="coerce")

    # Keep regular readings only
    df = df[df["Description"] == "REGULAR"]

    # Sort before diff
    df = df.sort_values(
        by=["C/A", "Unit", "SCP", "Datetime"]
    ).reset_index(drop=True)

    # Compute interval traffic
    df["Entry_Diff"] = df.groupby(["C/A", "Unit", "SCP"])["Entries"].diff()
    df["Exit_Diff"] = df.groupby(["C/A", "Unit", "SCP"])["Exits"].diff()

    # Remove negatives
    df.loc[df["Entry_Diff"] < 0, "Entry_Diff"] = None
    df.loc[df["Exit_Diff"] < 0, "Exit_Diff"] = None

    # Remove extreme spikes
    df.loc[df["Entry_Diff"] > 10000, "Entry_Diff"] = None
    df.loc[df["Exit_Diff"] > 10000, "Exit_Diff"] = None

    print("Data loaded and cleaned.\n")
    return df
