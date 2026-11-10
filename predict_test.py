"""Send example policies to the running service and print the answers.

    uv run python predict_test.py                       # local / Docker (port 9696)
    uv run python predict_test.py --url http://localhost:30096   # kind cluster
"""

import argparse

import requests

POLICIES = {
    "19-year-old, new car, penalised record, Paris region": {
        "Area": "E", "VehPower": 7, "VehAge": 0, "DrivAge": 19, "BonusMalus": 120,
        "VehBrand": "B1", "VehGas": "Regular", "Density": 4000, "Region": "R11"},
    "35-year-old, average city driver": {
        "Area": "C", "VehPower": 6, "VehAge": 5, "DrivAge": 35, "BonusMalus": 60,
        "VehBrand": "B2", "VehGas": "Regular", "Density": 800, "Region": "R82"},
    "55-year-old, best record, rural": {
        "Area": "A", "VehPower": 5, "VehAge": 8, "DrivAge": 55, "BonusMalus": 50,
        "VehBrand": "B12", "VehGas": "Diesel", "Density": 30, "Region": "R24"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9696")
    url = parser.parse_args().url.rstrip("/")

    print("GET /health ->", requests.get(f"{url}/health", timeout=10).json())
    for name, policy in POLICIES.items():
        r = requests.post(f"{url}/predict", json=policy, timeout=10)
        r.raise_for_status()
        print(f"\n{name}\n  -> {r.json()}")


if __name__ == "__main__":
    main()
