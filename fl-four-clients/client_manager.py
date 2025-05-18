import argparse
import subprocess
import json
import os
import sys
import signal
from pathlib import Path



PID_FILE = Path("temp_client_pids.json")
NUM_CLIENTS_DEFAULT = 3
NUM_MAX_CLIENTS = 4
EXECUTION_TEMPLATE = "from client_app import start_flower_client; start_flower_client({i}, {is_test_mode!r})"
TERMINAL_LOGGER_NAME_TEMPLATE = "terminal_client_{i}.log"




# -------------------------
# 1. Start
# -------------------------

def _start_clients(num_clients: int, is_test_mode: bool=False):
    processes = []
    folder_name = "logs"
    os.makedirs(folder_name, exist_ok=True)

    for i in range(1, num_clients +1):
        execution = EXECUTION_TEMPLATE.format(i=i, is_test_mode=is_test_mode)
        terminal_logger_name = TERMINAL_LOGGER_NAME_TEMPLATE.format(i=i)
        
        terminal_log_path = os.path.join(folder_name, terminal_logger_name)
        terminal_log_file = open(terminal_log_path, "w")
    
        proc = subprocess.Popen(
            ["python", "-c", execution],
            stdout=terminal_log_file, stderr=subprocess.STDOUT
        )
        processes.append({'pid': proc.pid, 'client_id': i})

        print(f"Started client {i} with PID {proc.pid}")

    with open(PID_FILE, "w") as f:
        json.dump(processes, f)




# -------------------------
# 2. Stop
# -------------------------

def _stop_clients():
    if not PID_FILE.exists():
        print("No running clients found.")
        return

    with open(PID_FILE) as f:
        processes = json.load(f)

    for p in processes:
        pid = p['pid']
        client_id = p['client_id']
        try:
            is_os_windows = os.name == 'nt'
            if is_os_windows:
                result = subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, text=True
                )
                if "not found" in result.stdout + result.stderr:
                    print(f"Client {client_id} (PID {pid}) already stopped.")
                    continue
            else:
                os.kill(pid, signal.SIGTERM)
            
            print(f"Stopped client {client_id} (PID {pid})")
            
        except Exception as e:
            print(f"Failed to stop client {client_id} (PID {pid}): {e}")

    PID_FILE.unlink()
    print("Cleaned PID history.")
    



# -------------------------
# 3. List
# -------------------------
    
def _is_process_running(pid: int) -> bool:
    response = False
    try:
        is_os_windows = os.name == 'nt'
        if is_os_windows:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True
            )
            response = str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            response = True
        
    except Exception:
        response = False
    
    return response


def _list_clients():
    if not PID_FILE.exists():
        print("No clients running.")
        return

    with open(PID_FILE) as f:
        processes = json.load(f)

    for p in processes:
        if _is_process_running(p['pid']):
            status = "🟢 running"
        else:
            status = "🔴 not running"
        print(f"Client {p['client_id']} — PID {p['pid']} — {status}")




# -------------------------
# 4. Main Execution
# -------------------------

def main():
    # python client_manager.py start <NUM_CLIENTS>
    # python client_manager.py stop 
    # python client_manager.py list 
    
    parser = argparse.ArgumentParser(description="Manage federated clients (start/stop/list)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start subcommand
    start_parser = subparsers.add_parser("start", help="Start a number of clients")
    start_parser.add_argument("num_clients", type=int, nargs="?", default=3, help="Number of clients to start")
    start_parser.add_argument("--test", action="store_true", help="Run clients in test mode")

    # stop subcommand
    subparsers.add_parser("stop", help="Stop all clients")

    # list subcommand
    subparsers.add_parser("list", help="List running clients")

    args = parser.parse_args()

    if args.command == "start":
        _start_clients(args.num_clients, is_test_mode=args.test)
    elif args.command == "stop":
        _stop_clients()
    elif args.command == "list":
        _list_clients()


if __name__ == "__main__":
    main()
