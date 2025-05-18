import subprocess
import json
import os
import sys
import signal
from pathlib import Path



PID_FILE = Path("temp_client_pids.json")
NUM_CLIENTS_DEFAULT = 3
NUM_MAX_CLIENTS = 4
# EXECUTION = f"from client_app import start_flower_client; start_flower_client({i})"
# TERMINAL_LOGGER_NAME = f"terminal_client_{i}.log"




# -------------------------
# 1. Start
# -------------------------

def start_clients(num_clients):
    processes = []
    folder_name = "logs"
    os.makedirs(folder_name, exist_ok=True)

    for i in range(1, num_clients +1):
        terminal_log_path = os.path.join(folder_name, f"terminal_client_{i}.log")
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




# -------------------------
# 2. Stop
# -------------------------

def stop_clients():
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
    
def is_process_running(pid):
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


def list_clients():
    if not PID_FILE.exists():
        print("No clients running.")
        return

    with open(PID_FILE) as f:
        processes = json.load(f)

    for p in processes:
        if is_process_running(p['pid']):
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
    
    if len(sys.argv) < 2:
        print("Usage: python client_manager.py [start N | stop | list]")
        return

    command = sys.argv[1]

    if command == "start":
        try:
            arg_num_clients = len(sys.argv)
            if arg_num_clients >= 2 and arg_num_clients <= NUM_MAX_CLIENTS:
                num = int(sys.argv[2])
            else:
                num = NUM_CLIENTS_DEFAULT
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
