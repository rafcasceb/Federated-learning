from client_app import start_flower_client

if __name__ == "__main__":  
    excel_file_name = "PI-CAI_3.xlsx" 
    temp_csv_file_name = "temp_database.csv"
    start_flower_client(excel_file_name, temp_csv_file_name, context=None)
