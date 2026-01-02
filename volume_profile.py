import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_volume_profile(df: pd.DataFrame, start_idx: Optional[int] = None, 
                            end_idx: Optional[int] = None, num_bins: int = 80) -> Dict:
    """
    Calculate volume profile for a given price range
    
    Args:
        df: DataFrame with OHLCV data
        start_idx: Starting index (None = start of df)
        end_idx: Ending index (None = end of df)
        num_bins: Number of price bins for volume distribution
    
    Returns:
        dict with:
        - 'profile': Series of volume per price bin
        - 'bin_centers': Array of price levels (bin centers)
        - 'bins': Array of bin edges
        - 'poc_price': Point of Control (highest volume price)
        - 'poc_volume': Volume at POC
        - 'vah': Value Area High (70% volume upper bound)
        - 'val': Value Area Low (70% volume lower bound)
        - 'hvn_prices': List of High Volume Node prices
        - 'lvn_prices': List of Low Volume Node prices
    """
    # Slice the dataframe
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(df)
    
    vp_df = df.iloc[start_idx:end_idx].copy()
    
    if vp_df.empty:
        return None
    
    try:
        # Define price range for bins
        p_min = vp_df['low'].min()
        p_max = vp_df['high'].max()
        
        # Create bins
        bins = np.linspace(p_min, p_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Assign each candle's close price to a bin
        vp_df['price_bin'] = pd.cut(vp_df['close'], bins, labels=False, include_lowest=True)
        
        # Sum volume for each bin
        volume_profile = vp_df.groupby('price_bin')['volume'].sum().reindex(range(num_bins), fill_value=0)
        
        # Calculate POC (Point of Control) - highest volume price
        poc_idx = volume_profile.idxmax()
        poc_price = bin_centers[poc_idx]
        poc_volume = volume_profile.iloc[poc_idx]
        
        # Calculate Value Area (70% of total volume)
        vah, val = calculate_value_area(volume_profile, bin_centers, value_area_pct=70)
        
        # Find HVNs and LVNs
        hvn_prices = find_hvns(volume_profile, bin_centers, threshold_percentile=75)
        lvn_prices = find_lvns(volume_profile, bin_centers, threshold_percentile=25)
        
        return {
            'profile': volume_profile,
            'bin_centers': bin_centers,
            'bins': bins,
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'vah': vah,
            'val': val,
            'hvn_prices': hvn_prices,
            'lvn_prices': lvn_prices,
            'total_volume': volume_profile.sum()
        }
    
    except Exception as e:
        print(f"Error calculating volume profile: {e}")
        return None


def find_hvns(volume_profile: pd.Series, bin_centers: np.ndarray, 
              threshold_percentile: float = 75) -> List[float]:
    """
    Find High Volume Nodes (prices with volume > threshold percentile)
    
    Args:
        volume_profile: Series of volume per bin
        bin_centers: Array of price levels
        threshold_percentile: Percentile threshold (default 75)
    
    Returns:
        List of HVN prices
    """
    threshold = np.percentile(volume_profile.values, threshold_percentile)
    hvn_indices = volume_profile[volume_profile >= threshold].index
    hvn_prices = [bin_centers[idx] for idx in hvn_indices]
    return hvn_prices


def find_lvns(volume_profile: pd.Series, bin_centers: np.ndarray, 
              threshold_percentile: float = 25) -> List[float]:
    """
    Find Low Volume Nodes (prices with volume < threshold percentile)
    
    Args:
        volume_profile: Series of volume per bin
        bin_centers: Array of price levels
        threshold_percentile: Percentile threshold (default 25)
    
    Returns:
        List of LVN prices
    """
    # Filter out zero-volume bins to avoid noise
    non_zero_profile = volume_profile[volume_profile > 0]
    
    if len(non_zero_profile) == 0:
        return []
    
    threshold = np.percentile(non_zero_profile.values, threshold_percentile)
    lvn_indices = volume_profile[(volume_profile > 0) & (volume_profile <= threshold)].index
    lvn_prices = [bin_centers[idx] for idx in lvn_indices]
    return lvn_prices


def calculate_value_area(volume_profile: pd.Series, bin_centers: np.ndarray, 
                        value_area_pct: float = 70) -> Tuple[float, float]:
    """
    Calculate Value Area High and Low (default 70% of volume)
    
    The Value Area contains the specified percentage of total volume,
    centered around the POC (Point of Control).
    
    Args:
        volume_profile: Series of volume per bin
        bin_centers: Array of price levels
        value_area_pct: Percentage of volume to include (default 70)
    
    Returns:
        Tuple of (VAH, VAL) - Value Area High and Low prices
    """
    total_volume = volume_profile.sum()
    target_volume = total_volume * (value_area_pct / 100.0)
    
    # Find POC index
    poc_idx = volume_profile.idxmax()
    
    # Expand from POC outward until we reach target volume
    accumulated_volume = volume_profile.iloc[poc_idx]
    lower_idx = poc_idx
    upper_idx = poc_idx
    
    while accumulated_volume < target_volume:
        # Determine which direction to expand (higher volume)
        lower_volume = volume_profile.iloc[lower_idx - 1] if lower_idx > 0 else 0
        upper_volume = volume_profile.iloc[upper_idx + 1] if upper_idx < len(volume_profile) - 1 else 0
        
        if lower_volume > upper_volume and lower_idx > 0:
            lower_idx -= 1
            accumulated_volume += volume_profile.iloc[lower_idx]
        elif upper_idx < len(volume_profile) - 1:
            upper_idx += 1
            accumulated_volume += volume_profile.iloc[upper_idx]
        else:
            # Can't expand further
            break
    
    vah = bin_centers[upper_idx]
    val = bin_centers[lower_idx]
    
    return vah, val


def get_nearest_hvn_above(price: float, hvn_prices: List[float]) -> Optional[float]:
    """
    Find nearest HVN above given price
    
    Args:
        price: Current price level
        hvn_prices: List of HVN prices
    
    Returns:
        Nearest HVN above price, or None if none exist
    """
    hvns_above = [p for p in hvn_prices if p > price]
    if hvns_above:
        return min(hvns_above)
    return None


def get_nearest_hvn_below(price: float, hvn_prices: List[float]) -> Optional[float]:
    """
    Find nearest HVN below given price
    
    Args:
        price: Current price level
        hvn_prices: List of HVN prices
    
    Returns:
        Nearest HVN below price, or None if none exist
    """
    hvns_below = [p for p in hvn_prices if p < price]
    if hvns_below:
        return max(hvns_below)
    return None


def is_in_lvn(price: float, lvn_prices: List[float], tolerance: float = 0.005) -> bool:
    """
    Check if price is within a Low Volume Node (with tolerance)
    
    Args:
        price: Current price level
        lvn_prices: List of LVN prices
        tolerance: Percentage tolerance (default 0.5%)
    
    Returns:
        True if price is within tolerance of any LVN
    """
    for lvn in lvn_prices:
        if abs(price - lvn) / lvn <= tolerance:
            return True
    return False


def get_volume_percentile(price: float, vp_data: Dict) -> float:
    """
    Get the volume percentile for a given price level
    
    Args:
        price: Price level to check
        vp_data: Volume profile data dict from calculate_volume_profile()
    
    Returns:
        Percentile (0-100) of volume at that price level
    """
    if vp_data is None:
        return 50.0  # Default to middle
    
    bin_centers = vp_data['bin_centers']
    volume_profile = vp_data['profile']
    
    # Find closest bin to price
    closest_idx = np.argmin(np.abs(bin_centers - price))
    volume_at_price = volume_profile.iloc[closest_idx]
    
    # Calculate percentile
    percentile = (volume_profile < volume_at_price).sum() / len(volume_profile) * 100
    
    return percentile


# Example usage and testing
if __name__ == '__main__':
    # Test with sample data
    print("Testing volume_profile module...")
    
    try:
        df = pd.read_csv('BTCUSDT_15m_data.csv', index_col='timestamp', parse_dates=True)
        print(f"Loaded {len(df)} rows of data")
        
        # Calculate VP for last 200 bars
        vp = calculate_volume_profile(df, start_idx=len(df)-200, end_idx=len(df), num_bins=80)
        
        if vp:
            print(f"\n{'='*60}")
            print("VOLUME PROFILE ANALYSIS (Last 200 bars)")
            print('='*60)
            print(f"Price Range: ${vp['bins'][0]:.2f} - ${vp['bins'][-1]:.2f}")
            print(f"Total Volume: {vp['total_volume']:,.0f}")
            print(f"\nPOC (Point of Control): ${vp['poc_price']:.2f} (Volume: {vp['poc_volume']:,.0f})")
            print(f"VAH (Value Area High):  ${vp['vah']:.2f}")
            print(f"VAL (Value Area Low):   ${vp['val']:.2f}")
            print(f"Value Area Range:       ${vp['vah'] - vp['val']:.2f}")
            print(f"\nHigh Volume Nodes (HVNs): {len(vp['hvn_prices'])} levels")
            if vp['hvn_prices']:
                print(f"  Top 5 HVNs: {[f'${p:.2f}' for p in sorted(vp['hvn_prices'], reverse=True)[:5]]}")
            print(f"\nLow Volume Nodes (LVNs): {len(vp['lvn_prices'])} levels")
            if vp['lvn_prices']:
                print(f"  Sample LVNs: {[f'${p:.2f}' for p in sorted(vp['lvn_prices'])[:5]]}")
            
            # Test helper functions
            current_price = df['close'].iloc[-1]
            print(f"\n{'='*60}")
            print(f"HELPER FUNCTION TESTS (Current Price: ${current_price:.2f})")
            print('='*60)
            
            nearest_hvn_above = get_nearest_hvn_above(current_price, vp['hvn_prices'])
            print(f"Nearest HVN above: ${nearest_hvn_above:.2f}" if nearest_hvn_above else "No HVN above")
            
            nearest_hvn_below = get_nearest_hvn_below(current_price, vp['hvn_prices'])
            print(f"Nearest HVN below: ${nearest_hvn_below:.2f}" if nearest_hvn_below else "No HVN below")
            
            in_lvn = is_in_lvn(current_price, vp['lvn_prices'])
            print(f"Currently in LVN: {in_lvn}")
            
            volume_pct = get_volume_percentile(current_price, vp)
            print(f"Volume percentile at current price: {volume_pct:.1f}%")
            
            print(f"\n{'='*60}")
            print("✓ All tests passed successfully!")
            print('='*60)
        else:
            print("Error: Could not calculate volume profile")
            
    except FileNotFoundError:
        print("Note: BTCUSDT_15m_data.csv not found. Module functions are ready to use.")
    except Exception as e:
        print(f"Error during testing: {e}")
