import argparse
import csv
import json
import math
import os
import subprocess
import time
from typing import Dict



NUM_RUNS = 3
NUM_CLIENTS = 4
SERVER_SCRIPT = "server.py"
CLIENT_MANAGER = "client_manager.py"
FOLDER = "logs"
METRICS_FILE = "final_aggr_metrics.json"
RESULTS_FILE = "experiment_results.csv"



def _run_one_experiment(run_id: int, is_test_mode: bool=False):
    print(f"Starting Run {run_id}")

    server_proc = subprocess.Popen(["python", SERVER_SCRIPT])
    print("Server started.")

    start_clients_cmd = ["python", CLIENT_MANAGER, "start", str(NUM_CLIENTS)]
    if is_test_mode:
        start_clients_cmd.append("--test")
    subprocess.run(start_clients_cmd)
    print("Clients started.")

    server_proc.wait()
    print("Server stopped.")
    
    time.sleep(5)
    subprocess.run(["python", CLIENT_MANAGER, "stop"])
    print("Clients stopped.")

    run_metrics = _extract_metrics()
    _log_results(run_id, run_metrics)

    print(f"Run {run_id} complete.\n")


def _extract_metrics() -> dict[str,float]:
    file_path = os.path.join(FOLDER, METRICS_FILE)
    file_exists = os.path.exists(file_path)
    
    if not file_exists:
        print(f"Metrics file not found: {file_path}")
        return {}

    with open(file_path, "r") as f:
        metrics = json.load(f)
    
    return metrics


def _log_results(run_id: int, metrics: Dict[str,float]):
    metrics_dict = {k: round(v, 2) for k, v in metrics.items()}
    metrics_dict["run_id"] = run_id
    fieldnames = ["run_id"] + list(metrics.keys())

    file_path = os.path.join(FOLDER, RESULTS_FILE)
    file_exists = os.path.exists(file_path)
    

    with open(file_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)
        
        
def _append_average_and_std_metrics(results_file_path: str):
    with open(results_file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader if row["run_id"].lower() not in {"avg", "std"}]

    if not rows:
        print("No data to process.")
        return

    metric_keys = [key for key in rows[0].keys() if key != "run_id"]
    metrics_data = {key: [] for key in metric_keys}

    for row in rows:
        for key in metric_keys:
            try:
                metrics_data[key].append(float(row[key]))
            except ValueError:
                print(f"Non-numeric value in '{key}'; skipping.")
    
    averages = {key: round(sum(vals) / len(vals), 2) for key, vals in metrics_data.items()}
    stddevs = {}
    for key, vals in metrics_data.items():
        if len(vals) > 1:
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            stddevs[key] = round(math.sqrt(variance), 2)
        else:
            stddevs[key] = 0.0

    # Add labels
    averages["run_id"] = "avg"
    stddevs["run_id"] = "std"

    # Append to file
    with open(results_file_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["run_id"] + metric_keys)
        writer.writerow(averages)
        writer.writerow(stddevs)

    print("Average and standard deviation metrics appended to CSV.")



def main():
    file_path = os.path.join(FOLDER, RESULTS_FILE)
    file_exists = os.path.exists(file_path)
    if file_exists:
        os.remove(file_path)
        
    parser = argparse.ArgumentParser(description="Run Federated Learning experiments")
    parser.add_argument("--test", action="store_true", help="Run in Test mode (some lightweight configs for non-deterministic reproducibility)")
    args = parser.parse_args()
    is_test_mode = args.test
        
    for run_id in range(1, NUM_RUNS + 1):
        _run_one_experiment(run_id, is_test_mode)
    
    _append_average_and_std_metrics(file_path)


if __name__ == "__main__":
    main()
    