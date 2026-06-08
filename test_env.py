"""Quick test to verify environment variables load correctly."""
import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    # Mask the key so the secret is never printed in full.
    masked = f"{api_key[:7]}...{api_key[-4:]}" if len(api_key) > 11 else "****"
    print(f"GOOGLE_API_KEY loaded successfully ({masked}).")
else:
    print("ERROR: GOOGLE_API_KEY not found. Check your .env file.")
    raise SystemExit(1)
