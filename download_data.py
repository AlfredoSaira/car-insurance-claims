"""Download the freMTPL2freq dataset from OpenML and save it to data/.

The dataset is public (OpenML id 41214) and needs no account. A compressed
copy is already committed in data/, so running this script is optional.

    uv run python download_data.py
"""

from pathlib import Path

from sklearn.datasets import fetch_openml

OPENML_ID = 41214
OUT = Path(__file__).parent / "data" / "freMTPL2freq.csv.gz"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    df = fetch_openml(data_id=OPENML_ID, as_frame=True).frame
    df["IDpol"] = df["IDpol"].astype(int)
    df["VehGas"] = df["VehGas"].astype(str).str.strip("'")
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"Saved {len(df):,} policies to {OUT}")


if __name__ == "__main__":
    main()
