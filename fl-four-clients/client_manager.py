import subprocess
import json
import os
import sys
import signal
from pathlib import Path



PID_FILE = Path("temp_client_pids.json")
NUM_CLIENTS_DEFAULT = 3


def start_clients(num_clients):
    processes = []
    folder_name = "logs"
    os.makedirs(folder_name, exist_ok=True)

    for i in range(1, num_clients +1):
        terminal_log_path = os.path.join(folder_name, f"termainal_client_{i}.log")
        terminal_log_file = open(terminal_log_path, "w")
    
        proc = subprocess.Popen([
            "python",
            "-c",
            f"from client_app import start_flower_client; start_flower_client({i})"
        ], stdout=terminal_log_file, stderr=subprocess.STDOUT)
        processes.append({'pid': proc.pid, 'client_id': i})

        print(f"Started client {i} with PID {proc.pid}")

    with open(PID_FILE, "w") as f:
        json.dump(processes, f)


def stop_clients():
    if not PID_FILE.exists():
        print("No running clients found.")
        return

    with open(PID_FILE) as f:
        processes = json.load(f)

    for p in processes:
        pid = p['pid']
        try:
            if os.name == 'nt':
                subprocess.call(["taskkill", "/F", "/PID", str(pid)])
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"Stopped client {p['client_id']} (PID {pid})")
        except Exception as e:
            print(f"Failed to stop client {p['client_id']} (PID {pid}): {e}")

    PID_FILE.unlink()


def list_clients():
    if not PID_FILE.exists():
        print("No clients running.")
        return

    with open(PID_FILE) as f:
        processes = json.load(f)

    print("Running Clients:")
    for p in processes:
        print(f"Client {p['client_id']} — PID {p['pid']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python client_manager.py [start N | stop | list]")
        return

    command = sys.argv[1]

    if command == "start":
        try:
            num = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_CLIENTS_DEFAULT
            start_clients(num)
        except ValueError:
            print("Please specify a valid number of clients.")
    elif command == "stop":
        stop_clients()
    elif command == "list":
        list_clients()
    else:
        print("Unknown command. Use start, stop, or list.")


if __name__ == "__main__":
    main()
