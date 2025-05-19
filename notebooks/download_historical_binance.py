import requests
import os
import hashlib
import zipfile
from datetime import datetime

# --- Configuration ---
SYMBOL = "BTCUSDT"  # Bitcoin against USDT. Change if you need a different pair (e.g., BTCBUSD)
YEAR = 2025
MONTH = 5
DAY = 16
INTERVAL = "1m"  # 1 minute klines
MARKET_TYPE = "spot"  # 'spot' or 'futures' (though futures has sub-types)
DATA_TYPE = "klines"  # 'klines', 'aggTrades', 'trades'
BASE_URL = "https://data.binance.vision/data"
OUTPUT_DIRECTORY = "./binance_data"  # Where to save downloaded files

# --- Helper Functions ---
def download_file(url, local_filename): # CORRECTED
    """Downloads a file from a URL to a local path."""
    print(f"Downloading {url} to {local_filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Will raise an HTTPError for bad responses (4XX or 5XX)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded {local_filename}")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Error: File not found (404) at {url}. Data for {YEAR}-{MONTH:02d}-{DAY:02d} might not be available yet.")
            print("Daily data typically becomes available the NEXT day.")
        else:
            print(f"HTTP Error during download: {e}")
        return False
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return False

def verify_checksum(zip_filepath, checksum_filepath): # CORRECTED
    """Verifies the SHA256 checksum of the downloaded file."""
    print(f"Verifying checksum for {zip_filepath} using {checksum_filepath}...")
    try:
        with open(checksum_filepath, 'r') as f:
            checksum_content = f.read().strip()
            # The checksum file usually contains "checksum  filename"
            expected_checksum = checksum_content.split()[0]

        hasher = hashlib.sha256()
        with open(zip_filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        calculated_checksum = hasher.hexdigest()

        if calculated_checksum == expected_checksum:
            print(f"Checksum VERIFIED: {calculated_checksum}")
            return True
        else:
            print(f"Checksum MISMATCH!")
            print(f"  Expected:   {expected_checksum}")
            print(f"  Calculated: {calculated_checksum}")
            return False
    except FileNotFoundError:
        print(f"Error: Checksum file {checksum_filepath} not found.")
        return False
    except Exception as e:
        print(f"Error during checksum verification: {e}")
        return False

def extract_zip(zip_filepath, extract_to_directory): # CORRECTED
    """Extracts a zip file to a specified directory."""
    print(f"Extracting {zip_filepath} to {extract_to_directory}...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_directory)
        # The CSV inside the zip usually has the same name as the zip, but with .csv
        # e.g., BTCUSDT-1m-2025-05-15.zip contains BTCUSDT-1m-2025-05-15.csv
        extracted_filename = os.path.splitext(os.path.basename(zip_filepath))[0] + ".csv"
        print(f"Successfully extracted. Main file: {os.path.join(extract_to_directory, extracted_filename)}")
        return True
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False

# --- Main Script ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Validate date (simple check, more robust needed for production)
    try:
        target_date = datetime(YEAR, MONTH, DAY)
        if target_date > datetime.now():
            print(f"Warning: The requested date {YEAR}-{MONTH:02d}-{DAY:02d} is in the future.")
            print("Data for this date will not exist.")
            # You might want to exit here if this is an error for your use case
            # exit()
    except ValueError:
        print(f"Error: Invalid date {YEAR}-{MONTH:02d}-{DAY:02d}")
        exit()

    # Construct filenames and URLs
    # For daily data, the format is SYMBOL-INTERVAL-YYYY-MM-DD.zip
    filename_base = f"{SYMBOL}-{INTERVAL}-{YEAR}-{MONTH:02d}-{DAY:02d}"
    zip_filename = f"{filename_base}.zip"
    checksum_filename = f"{zip_filename}.CHECKSUM"

    # URL path: /data/{market_type}/daily/{data_type}/{SYMBOL}/{INTERVAL}/{zip_filename}
    data_url = f"{BASE_URL}/{MARKET_TYPE}/daily/{DATA_TYPE}/{SYMBOL}/{INTERVAL}/{zip_filename}"
    checksum_url = f"{data_url}.CHECKSUM"

    local_zip_filepath = os.path.join(OUTPUT_DIRECTORY, zip_filename)
    local_checksum_filepath = os.path.join(OUTPUT_DIRECTORY, checksum_filename)

    print(f"--- Preparing to download Bitcoin {INTERVAL} Klines (Spot) ---")
    print(f"Symbol: {SYMBOL}")
    print(f"Date: {YEAR}-{MONTH:02d}-{DAY:02d}")
    print(f"Data URL: {data_url}")
    print(f"Checksum URL: {checksum_url}")
    print("---")
    
    print("Important Note: Daily data typically becomes available the *next* day.")
    print(f"So, data for {YEAR}-{MONTH:02d}-{DAY:02d} is expected to be available on {YEAR}-{MONTH:02d}-{DAY+1:02d} (or later).")
    print("---")


    # 1. Download the data ZIP file
    if not download_file(data_url, local_zip_filepath): # CORRECTED CALL
        print("Failed to download data file. Exiting.")
        exit()

    # 2. Download the CHECKSUM file
    if not download_file(checksum_url, local_checksum_filepath): # CORRECTED CALL
        print("Failed to download checksum file. Integrity cannot be verified. Exiting.")
        # Optionally, you could proceed without checksum verification if you choose
        exit()

    # 3. Verify Checksum
    if not verify_checksum(local_zip_filepath, local_checksum_filepath): # CORRECTED CALL
        print("Checksum verification failed. The downloaded data might be corrupted or incomplete.")
        # Decide how to handle this: delete the file, retry, or warn the user
        # For now, we'll exit
        # os.remove(local_zip_filepath) # Optionally delete
        # os.remove(local_checksum_filepath) # Optionally delete
        exit()
    else:
        print("Data integrity confirmed via checksum.")

    # 4. (Optional) Extract the ZIP file
    extract_choice = input(f"Do you want to extract {zip_filename}? (y/N): ").strip().lower()
    if extract_choice == 'y':
        # Extract into a sub-folder named after the zip file (without .zip) for neatness
        extract_folder = os.path.join(OUTPUT_DIRECTORY, filename_base)
        os.makedirs(extract_folder, exist_ok=True)
        if extract_zip(local_zip_filepath, extract_folder): # CORRECTED CALL
            print(f"Data extracted to: {extract_folder}")
        else:
            print("Failed to extract the zip file.")
    else:
        print(f"ZIP file saved at {local_zip_filepath}. Checksum file at {local_checksum_filepath}")

    print("--- Script Finished ---")