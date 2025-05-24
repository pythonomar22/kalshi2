# test_kalshi_api.py
import httpx
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from datetime import datetime, timezone, timedelta # For time filtering

# --- Load .env for API credentials ---
dotenv_path = Path(__file__).parent / '.env' # Assumes .env is in the same dir or parent if this is in a subdir
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent.parent / '.env' # Try one level up if script is in a subdir

load_dotenv(dotenv_path=dotenv_path)

# --- Kalshi API Configuration (copied from paper/config.py for standalone use) ---
IS_DEMO_MODE = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"

if IS_DEMO_MODE:
    print("KALSHI: Testing in DEMO mode.")
    KALSHI_API_KEY_ID = os.getenv("KALSHI_DEMO_API_KEY_ID")
    KALSHI_PRIVATE_KEY_PATH_STR = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH")
    KALSHI_API_BASE_URL = os.getenv("KALSHI_DEMO_API_BASE_URL", "https://demo-api.kalshi.co")
else:
    print("KALSHI: Testing in PRODUCTION mode.")
    KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
    KALSHI_PRIVATE_KEY_PATH_STR = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    KALSHI_API_BASE_URL = os.getenv("KALSHI_PROD_API_BASE_URL", "https://api.elections.kalshi.com")

if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH_STR:
    print("CRITICAL: Kalshi API Key ID or Private Key Path not found in .env.")
    exit(1)

KALSHI_PRIVATE_KEY_PATH = Path(KALSHI_PRIVATE_KEY_PATH_STR).expanduser().resolve()
if not KALSHI_PRIVATE_KEY_PATH.exists():
    print(f"CRITICAL: Kalshi private key file not found at: {KALSHI_PRIVATE_KEY_PATH}")
    exit(1)

# --- Auth Functions (copied from paper/utils.py) ---
def load_private_key(file_path: Path) -> rsa.RSAPrivateKey | None:
    try:
        with open(file_path, "rb") as key_file:
            return serialization.load_pem_private_key(key_file.read(), password=None)
    except Exception as e:
        print(f"Error loading private key from {file_path}: {e}")
        return None

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str | None:
    message = text.encode('utf-8')
    try:
        signature_bytes = private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error during PSS signing: {e}")
        return None

def get_kalshi_auth_headers_for_test(method: str, path: str, private_key: rsa.RSAPrivateKey, key_id: str) -> dict | None:
    if not private_key or not key_id: return None
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

async def test_get_markets():
    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH)
    if not private_key:
        return

    endpoint = "/trade-api/v2/markets"
    
    # --- Parameters to test ---
    # Test 1: Just status='open'
    # params_to_test = {"status": "open", "limit": 10}
    
    # Test 2: status='open' AND series_ticker='KXBTCD'
    params_to_test = {"status": "open", "series_ticker": "KXBTCD", "limit": 50}
    
    # Test 3: Just series_ticker='KXBTCD' (to see all its markets regardless of status)
    # params_to_test = {"series_ticker": "KXBTCD", "limit": 50}

    # Test 4: Specific event_ticker (if you know one is active)
    # Example: event_ticker for "Bitcoin price today at 9pm EDT?" might be KXBTCD-25MAY2321
    # params_to_test = {"event_ticker": "KXBTCD-25MAY2321", "limit": 10}


    print(f"Querying: {KALSHI_API_BASE_URL}{endpoint} with params: {params_to_test}")
    auth_headers = get_kalshi_auth_headers_for_test("GET", endpoint, private_key, KALSHI_API_KEY_ID)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{KALSHI_API_BASE_URL}{endpoint}", headers=auth_headers, params=params_to_test, timeout=20.0)
            print(f"\nResponse Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                print(f"Found {len(markets)} markets matching criteria on this page.")
                if markets:
                    print("First few markets found:")
                    for i, market in enumerate(markets[:5]): # Print first 5
                        print(f"  --- Market {i+1} ---")
                        print(f"    Ticker: {market.get('ticker')}")
                        print(f"    Event Ticker: {market.get('event_ticker')}")
                        print(f"    Series Ticker: {market.get('series_ticker')}")
                        print(f"    Title: {market.get('title')}")
                        print(f"    Status: {market.get('status')}")
                        print(f"    Open Time: {market.get('open_time')}")
                        print(f"    Close Time: {market.get('close_time')}")
                        print(f"    Strike: {market.get('strike')}")
                        # Filter for relevance here if needed for testing bot's logic
                        now_utc = datetime.now(timezone.utc)
                        open_time_dt = datetime.fromisoformat(market['open_time'].replace('Z', '+00:00'))
                        close_time_dt = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))
                        time_until_close = close_time_dt - now_utc

                        is_relevant_for_bot = (
                            market.get('series_ticker') == "KXBTCD" and
                            market.get('status') == 'open' and
                            open_time_dt <= now_utc < close_time_dt and
                            timedelta(minutes=1 + 1) < time_until_close < timedelta(hours=2, minutes=30) # Using MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION = 1
                        )
                        print(f"    Is relevant by bot's time window criteria: {is_relevant_for_bot}")

                if data.get("cursor"):
                    print(f"Next page cursor: {data.get('cursor')}")
                else:
                    print("No more pages.")
            else:
                print(f"Error Response Content: {response.text}")

        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")

async def main():
    await test_get_markets()

if __name__ == "__main__":
    asyncio.run(main())