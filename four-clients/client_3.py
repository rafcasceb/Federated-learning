from client_app import start_flower_client


CLIENT_ID = 3

if __name__ == "__main__":  
    excel_file_name = f"PI-CAI_3__part{CLIENT_ID}.xlsx" 
    temp_csv_file_name = f"temp_database_{CLIENT_ID}.csv"
    logger_name = f"client{CLIENT_ID}.log"
    
    start_flower_client(CLIENT_ID, excel_file_name, temp_csv_file_name, logger_name, context=None)
    