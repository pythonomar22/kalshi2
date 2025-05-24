# /paper/utils.py

import joblib
import json
import logging
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import base64
import time
import httpx 
import zipfile
import os 
import shutil 
from typing import Dict, Optional, List, Any, Tuple # Ensure Optional and other needed types are here

import paper.config as cfg 

logger = logging.getLogger("paper_utils")

def load_model_and_dependencies():
    # ... (same as before)
    try:
        model = joblib.load(cfg.MODEL_PATH)
        scaler = joblib.load(cfg.SCALER_PATH)
        with open(cfg.FEATURE_NAMES_PATH, 'r') as f:
            feature_names = json.load(f)
        logger.info(f"Successfully loaded model from {cfg.MODEL_PATH}, scaler, and {len(feature_names)} feature names.")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        logger.error(f"Error loading model/scaler/features: {e}. Check paths in config.py. Paths: M={cfg.MODEL_PATH}, S={cfg.SCALER_PATH}, F={cfg.FEATURE_NAMES_PATH}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred loading model components: {e}", exc_info=True)
        raise

def load_private_key(file_path_str: str) -> Optional[rsa.RSAPrivateKey]: # Added Optional here too
    # ... (same as before)
    try:
        expanded_path = Path(file_path_str).expanduser().resolve()
        if not expanded_path.exists():
            logger.error(f"Private key file does not exist at resolved path: {expanded_path}")
            return None
        with open(expanded_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(key_file.read(), password=None)
        return private_key
    except FileNotFoundError: 
        logger.error(f"Private key file not found: {file_path_str}")
        return None
    except Exception as e:
        logger.error(f"Error loading private key from {file_path_str}: {e}")
        return None


def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> Optional[str]: # Added Optional
    # ... (same as before)
    if not private_key:
        logger.error("Private key not available for signing.")
        return None
    message = text.encode('utf-8')
    try:
        signature_bytes = private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Error during PSS signing: {e}")
        return None

def get_kalshi_auth_headers(method: str, path: str, private_key: rsa.RSAPrivateKey, key_id: str) -> Optional[Dict[str, str]]: # Added Optional and Dict
    # ... (same as before)
    if not private_key or not key_id:
        logger.error("Cannot create Kalshi auth headers: Missing private key or API key ID.")
        return None
    timestamp_ms_str = str(int(time.time() * 1000))
    path_to_sign = path if path.startswith('/') else '/' + path
    message_to_sign = timestamp_ms_str + method.upper() + path_to_sign
    signature = sign_pss_text(private_key, message_to_sign)
    if signature is None: return None
    return {
        'accept': 'application/json',
        'Content-Type': 'application/json', 
        'KALSHI-ACCESS-KEY': key_id,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp_ms_str
    }

def get_kalshi_ws_auth_headers(private_key: rsa.RSAPrivateKey, key_id: str) -> Optional[Dict[str, str]]: # Added Optional and Dict
    # ... (same as before)
    if not private_key or not key_id:
        logger.error("Cannot create Kalshi WebSocket auth headers: Missing private key or API key ID.")
        return None
    ws_path_for_signing = "/trade-api/ws/v2" 
    method = "GET" 
    timestamp_ms_str = str(int(time.time() * 1000))
    message_to_sign = timestamp_ms_str + method.upper() + ws_path_for_signing
    signature = sign_pss_text(private_key, message_to_sign)
    if signature is None: return None
    return {
        'KALSHI-ACCESS-KEY': key_id,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp_ms_str
    }

# --- New Helper Functions for Binance Data Pre-fill ---
async def download_file_async(client: httpx.AsyncClient, url: str, local_path: Path) -> bool:
    logger.info(f"Downloading {url} to {local_path}...")
    try:
        async with client.stream("GET", url, timeout=60.0) as response: 
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
        logger.info(f"Successfully downloaded {local_path}")
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"File not found (404) at {url}. Data might not be available for this day/period yet.")
        else:
            logger.error(f"HTTP Error during download: {e.response.status_code} for {url} - {e.response.text}") # Log response text
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download of {url}: {e}", exc_info=True)
        return False

def extract_zip_file(zip_filepath: Path, extract_to_dir: Path) -> Optional[Path]:
    logger.info(f"Extracting {zip_filepath} to {extract_to_dir}...")
    extract_to_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            csv_filename_in_zip = None
            for member_name in zip_ref.namelist():
                if member_name.lower().endswith('.csv'):
                    csv_filename_in_zip = member_name
                    break
            if not csv_filename_in_zip:
                logger.error(f"No CSV file found within ZIP: {zip_filepath.name}")
                return None
            
            zip_ref.extractall(extract_to_dir)
            extracted_csv_path = extract_to_dir / csv_filename_in_zip
            if extracted_csv_path.exists():
                logger.info(f"Successfully extracted. Main CSV: {extracted_csv_path}")
                return extracted_csv_path
            else:
                logger.error(f"Extraction seemed successful but CSV not found at: {extracted_csv_path}")
                return None
    except zipfile.BadZipFile:
        logger.error(f"Bad ZIP file for {zip_filepath}. It might be corrupted or incomplete.")
        return None
    except Exception as e:
        logger.error(f"Error during extraction of {zip_filepath}: {e}", exc_info=True)
        return None

def cleanup_temp_dir(temp_dir: Path):
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary prefill directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")