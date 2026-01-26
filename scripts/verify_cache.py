
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import get_session_dataframe, prepare_training_data
from src.utils import setup_logging

def main():
    setup_logging()
    db_path = Path("data/processed/cert.duckdb")
    
    print("Loading dataframe...")
    df = get_session_dataframe(db_path)
    
    print("\n--- Run 1: Should calculate and save cache (Slow) ---")
    start = time.time()
    _ = prepare_training_data(df, sequence_length=100)
    duration1 = time.time() - start
    print(f"Run 1 took: {duration1:.2f} seconds")
    
    print("\n--- Run 2: Should load from cache (Fast) ---")
    start = time.time()
    _ = prepare_training_data(df, sequence_length=100)
    duration2 = time.time() - start
    print(f"Run 2 took: {duration2:.2f} seconds")
    
    speedup = duration1 / duration2 if duration2 > 0 else 0
    print(f"\nSpeedup: {speedup:.1f}x faster")
    
    if duration2 < duration1 and speedup >= 2.0:
        print("✅ SUCCESS: Caching is working! (Run 2 was significantly faster)")
    elif duration2 < duration1:
        print("⚠️ PARTIAL: Caching works but speedup is minimal (large cache file)")
    else:
        print("❌ FAILURE: Caching might not be working.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
