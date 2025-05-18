import csv
import json
import math
import os
import subprocess
import time
from typing import Dict



NUM_RUNS = 2
NUM_CLIENTS = 4
SERVER_SCRIPT = "server.py"
CLIENT_MANAGER = "client_manager.py"
METRICS_FOLDER = "logs"
METRICS_FILE = "final_aggr_metrics.json"
RESULTS_FILE = "experiment_results.csv"



def run_one_experiment(run_id):
    print(f"Starting Run {run_id}")

    server_proc = subprocess.Popen(["python", SERVER_SCRIPT])
    print("Server started.")

    subprocess.run(["python", CLIENT_MANAGER, "start", str(NUM_CLIENTS)])
    print("Clients started.")

    server_proc.wait()
    print("Server stopped.")
    
    time.sleep(5)
    subprocess.run(["python", CLIENT_MANAGER, "stop"])
    print("Clients stopped.")

    run_metrics = extract_metrics()
    log_results(run_id, run_metrics)

    print(f"Run {run_id} complete.\n")


def extract_metrics() -> dict[str,float]:
    metrics_file = os.path.join(METRICS_FOLDER, METRICS_FILE)
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return {}

    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    return metrics


def log_results(run_id: int, metrics: Dict[str,float]):
    metrics_dict = {k: round(v, 2) for k, v in metrics.items()}
    metrics_dict["run_id"] = run_id
    fieldnames = ["run_id"] + list(metrics.keys())

    file_exists = os.path.isfile(RESULTS_FILE)  # ver si lo meto en la carpeta logs

    with open(RESULTS_FILE, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)
        
        
def append_average_and_std_metrics(results_file: str):
    with open(results_file, "r", newline="") as csvfile:
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
    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["run_id"] + metric_keys)
        writer.writerow(averages)
        writer.writerow(stddevs)

    print("Average and standard deviation metrics appended to CSV.")



def main():
    file_exists = os.path.isfile(RESULTS_FILE)  # ver si lo meto en la carpeta logs
    if file_exists:
        os.remove(RESULTS_FILE)
        
    for run_id in range(1, NUM_RUNS + 1):
        run_one_experiment(run_id)
    
    append_average_and_std_metrics(RESULTS_FILE)


if __name__ == "__main__":
    main()
    