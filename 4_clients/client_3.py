from client_app import start_flower_client


CLIENT_NUMBER = 3

if __name__ == "__main__":  
    excel_file_name = f"PI-CAI_3__part{CLIENT_NUMBER}.xlsx" 
    temp_csv_file_name = f"temp_database_{CLIENT_NUMBER}.csv"
    logger_name = f"client{CLIENT_NUMBER}.log"
    
    start_flower_client(excel_file_name, temp_csv_file_name, logger_name, context=None)
    