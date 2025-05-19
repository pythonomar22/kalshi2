import requests
import os
import hashlib
import zipfile
import time
from datetime import datetime, timedelta, date

# --- Configuration ---
SYMBOL = "BTCUSDT"  # Bitcoin against USDT.
INTERVAL = "1m"  # 1 minute klines
MARKET_TYPE = "spot"
DATA_TYPE = "klines"
BASE_URL = "https://data.binance.vision/data"
OUTPUT_DIRECTORY = "./binance_data"  # Where to save downloaded files

# --- Date Range Configuration ---
# Inclusive start and end dates
START_DATE_STR = "2024-01-01" # YYYY-MM-DD
# Default to yesterday, as today's data might not be fully available.
# You can set a specific end date like "2024-01-05" for testing, or "2025-05-16" for your full range.
DEFAULT_END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
END_DATE_STR = os.getenv("BINANCE_DOWNLOAD_END_DATE", DEFAULT_END_DATE) # YYYY-MM-DD

AUTO_EXTRACT_ZIP = True # Set to True to automatically extract CSVs
DOWNLOAD_DELAY_SECONDS = 1 # Small delay between file downloads

# --- Helper Functions ---
def download_file(url, local_filename):
    """Downloads a file from a URL to a local path."""
    print(f"Downloading {url} to {local_filename}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r: # Added timeout
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded {local_filename}")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Error: File not found (404) at {url}. Data might not be available yet or does not exist.")
        else:
            print(f"HTTP Error during download: {e.response.status_code} for {url}")
        return False
    except requests.exceptions.RequestException as e: # Catch other request errors (timeout, connection)
        print(f"Request error during download: {e} for {url}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download of {url}: {e}")
        return False

def verify_checksum(zip_filepath, checksum_filepath):
    """Verifies the SHA256 checksum of the downloaded file."""
    print(f"Verifying checksum for {zip_filepath} using {checksum_filepath}...")
    if not os.path.exists(zip_filepath):
        print(f"Error: Zip file {zip_filepath} not found for checksum verification.")
        return False
    if not os.path.exists(checksum_filepath):
        print(f"Error: Checksum file {checksum_filepath} not found.")
        return False
    try:
        with open(checksum_filepath, 'r') as f:
            checksum_content = f.read().strip()
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
    except Exception as e:
        print(f"Error during checksum verification: {e}")
        return False

def extract_zip(zip_filepath, extract_to_directory):
    """Extracts a zip file to a specified directory."""
    print(f"Extracting {zip_filepath} to {extract_to_directory}...")
    if not os.path.exists(zip_filepath):
        print(f"Error: Zip file {zip_filepath} not found for extraction.")
        return False
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_directory)
        extracted_filename = os.path.splitext(os.path.basename(zip_filepath))[0] + ".csv"
        print(f"Successfully extracted. Main file: {os.path.join(extract_to_directory, extracted_filename)}")
        return True
    except zipfile.BadZipFile:
        print(f"Error: Bad ZIP file for {zip_filepath}. It might be corrupted or incomplete.")
        return False
    except Exception as e:
        print(f"Error during extraction of {zip_filepath}: {e}")
        return False

def process_single_day(year, month, day):
    """Downloads, verifies, and extracts data for a single day."""
    print(f"\n--- Processing data for {year}-{month:02d}-{day:02d} ---")

    filename_base = f"{SYMBOL}-{INTERVAL}-{year}-{month:02d}-{day:02d}"
    zip_filename = f"{filename_base}.zip"
    checksum_filename = f"{zip_filename}.CHECKSUM"

    data_url = f"{BASE_URL}/{MARKET_TYPE}/daily/{DATA_TYPE}/{SYMBOL}/{INTERVAL}/{zip_filename}"
    checksum_url = f"{data_url}.CHECKSUM"

    local_zip_filepath = os.path.join(OUTPUT_DIRECTORY, zip_filename)
    local_checksum_filepath = os.path.join(OUTPUT_DIRECTORY, checksum_filename)
    
    # Create specific output directory for this day's CSV if extracting
    # This keeps CSVs like: ./binance_data/BTCUSDT-1m-YYYY-MM-DD/BTCUSDT-1m-YYYY-MM-DD.csv
    # And Zips like: ./binance_data/BTCUSDT-1m-YYYY-MM-DD.zip
    extract_folder = os.path.join(OUTPUT_DIRECTORY, filename_base) 

    # Skip if CSV already exists and successfully extracted
    expected_csv_path = os.path.join(extract_folder, f"{filename_base}.csv")
    if AUTO_EXTRACT_ZIP and os.path.exists(expected_csv_path) and os.path.getsize(expected_csv_path) > 0:
        print(f"CSV file {expected_csv_path} already exists. Skipping download for {year}-{month:02d}-{day:02d}.")
        return True # Indicate success for this day

    # 1. Download the data ZIP file
    if not os.path.exists(local_zip_filepath) or os.path.getsize(local_zip_filepath) == 0:
        if not download_file(data_url, local_zip_filepath):
            print(f"Failed to download data file for {year}-{month:02d}-{day:02d}.")
            if os.path.exists(local_zip_filepath): os.remove(local_zip_filepath) # Clean up empty/failed file
            return False
    else:
        print(f"ZIP file {local_zip_filepath} already exists.")

    # 2. Download the CHECKSUM file
    if not os.path.exists(local_checksum_filepath) or os.path.getsize(local_checksum_filepath) == 0:
        if not download_file(checksum_url, local_checksum_filepath):
            print(f"Failed to download checksum file for {year}-{month:02d}-{day:02d}.")
            # Proceeding without checksum is risky, but some very old checksums might be missing.
            # For now, we'll return False to indicate an issue.
            return False
    else:
        print(f"Checksum file {local_checksum_filepath} already exists.")

    # 3. Verify Checksum
    if not verify_checksum(local_zip_filepath, local_checksum_filepath):
        print(f"Checksum verification failed for {year}-{month:02d}-{day:02d}.")
        # Optionally remove the corrupted zip
        # os.remove(local_zip_filepath)
        return False
    else:
        print("Data integrity confirmed via checksum.")

    # 4. Extract the ZIP file if AUTO_EXTRACT_ZIP is True
    if AUTO_EXTRACT_ZIP:
        os.makedirs(extract_folder, exist_ok=True)
        if not extract_zip(local_zip_filepath, extract_folder):
            print(f"Failed to extract the zip file for {year}-{month:02d}-{day:02d}.")
            return False
        else:
            print(f"Data extracted to: {extract_folder}")
    else:
        print(f"ZIP file saved at {local_zip_filepath}. Auto-extraction is disabled.")
    
    return True


# --- Main Script ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    try:
        start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d").date()
        end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Invalid START_DATE_STR or END_DATE_STR format. Please use YYYY-MM-DD.")
        exit()

    if start_date > end_date:
        print(f"Error: Start date ({start_date}) is after end date ({end_date}). Nothing to download.")
        exit()

    print(f"--- Preparing to download Binance {INTERVAL} Klines ({MARKET_TYPE}) ---")
    print(f"Symbol: {SYMBOL}")
    print(f"Date Range: {start_date.isoformat()} to {end_date.isoformat()} (inclusive)")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    print(f"Auto-extract ZIP: {AUTO_EXTRACT_ZIP}")
    print("---")
    print("Important Note: Daily data for a specific day (e.g., YYYY-MM-DD) typically becomes available on YYYY-MM-(DD+1).")
    print("If END_DATE_STR is set to today or yesterday, some files might not be available yet.")
    print("---")

    current_date = start_date
    days_processed_successfully = 0
    days_failed = 0
    total_days = (end_date - start_date).days + 1

    while current_date <= end_date:
        if current_date >= datetime.now().date():
            print(f"Skipping {current_date.isoformat()}: Date is today or in the future, data likely unavailable.")
            current_date += timedelta(days=1)
            total_days -=1 # Adjust total days to reflect skipped future dates
            continue
            
        if process_single_day(current_date.year, current_date.month, current_date.day):
            days_processed_successfully += 1
        else:
            days_failed +=1
            print(f"!!! Processing failed for {current_date.isoformat()} !!!")

        current_date += timedelta(days=1)
        if current_date <= end_date: # Only sleep if there are more days to process
            print(f"Waiting {DOWNLOAD_DELAY_SECONDS}s before next download...")
            time.sleep(DOWNLOAD_DELAY_SECONDS)

    print("\n--- Download Range Processing Finished ---")
    print(f"Total days in range attempted (excluding future dates): {total_days}")
    print(f"Successfully processed/found: {days_processed_successfully} days")
    print(f"Failed to process: {days_failed} days")
    if days_failed > 0:
        print("Check logs for errors for the failed dates.")
    print("CSVs (if extracted) are in subdirectories named like 'BTCUSDT-1m-YYYY-MM-DD' within the output directory.")