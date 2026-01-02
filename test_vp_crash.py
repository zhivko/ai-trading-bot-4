import pandas as pd
import numpy as np
from volume_profile import calculate_volume_profile

def test_vp():
    print("Testing Volume Profile Calculation...")
    
    # Create dummy data
    dates = pd.date_range(start='2024-01-01', periods=300, freq='1h')
    data = {
        'open': np.random.randn(300) + 100,
        'high': np.random.randn(300) + 102,
        'low': np.random.randn(300) + 98,
        'close': np.random.randn(300) + 100,
        'volume': np.random.rand(300) * 1000
    }
    df = pd.DataFrame(data, index=dates)
    
    try:
        vp = calculate_volume_profile(df, num_bins=80)
        print("Success!")
        print("Clusters:", vp.get('cluster_prices'))
    except Exception as e:
        print("CRASHED:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vp()
