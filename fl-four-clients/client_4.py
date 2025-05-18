from client_app import start_flower_client


CLIENT_ID = 4
is_test_run = True

if __name__ == "__main__":
    start_flower_client(CLIENT_ID, is_test_run)
