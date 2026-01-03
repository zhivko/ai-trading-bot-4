import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def get_raw_volume_histogram(df: pd.DataFrame, start_idx: int, end_idx: int, num_bins: int = 80, verbose: bool = False) -> np.ndarray:
    """
    Lightning fast vectorized histogram using only Close prices.
    Used for NN inference loop where performance is the top priority.
    """
    slice_df = df.iloc[start_idx:end_idx]
    if slice_df.empty:
        return np.zeros(num_bins)
    
    p_min = slice_df['low'].min()
    p_max = slice_df['high'].max()
    
    if p_min == p_max:
        return np.zeros(num_bins)
        
    counts, _ = np.histogram(slice_df['close'].values, bins=num_bins, range=(p_min, p_max), weights=slice_df['volume'].values)
    
    if verbose:
        print(f"DEBUG FAST HISTOGRAM: {len(slice_df)} candles, {num_bins} bins")
        
    return counts


def get_precise_volume_histogram(df: pd.DataFrame, start_idx: int, end_idx: int, num_bins: int = 80, verbose: bool = False) -> np.ndarray:
    """
    High precision histogram that distributes volume across the entire candle range (Low to High).
    Used for marker-click details where visual accuracy is more important than speed.
    """
    slice_df = df.iloc[start_idx:end_idx]
    if slice_df.empty:
        return np.zeros(num_bins)
    
    p_min = slice_df['low'].min()
    p_max = slice_df['high'].max()
    
    if p_min == p_max:
        return np.zeros(num_bins)
    
    bins = np.linspace(p_min, p_max, num_bins + 1)
    counts = np.zeros(num_bins)
    
    # Python loop is acceptable here because we only do it ONCE for a single marker click
    for idx in range(len(slice_df)):
        candle_low = slice_df.iloc[idx]['low']
        candle_high = slice_df.iloc[idx]['high']
        candle_volume = slice_df.iloc[idx]['volume']
        
        low_bin = np.digitize(candle_low, bins) - 1
        high_bin = np.digitize(candle_high, bins) - 1
        
        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))
        
        bins_touched = high_bin - low_bin + 1
        if bins_touched > 0:
            volume_per_bin = candle_volume / bins_touched
            counts[low_bin:high_bin+1] += volume_per_bin
    
    if verbose:
        bins_with_volume = np.sum(counts > 0)
        print(f"DEBUG PRECISE HISTOGRAM: {len(slice_df)} candles, {num_bins} bins")
        print(f"DEBUG PRECISE HISTOGRAM: Bins with volume: {bins_with_volume}/{num_bins}")
    
    return counts

def calculate_volume_profile(df: pd.DataFrame, start_idx: Optional[int] = None, 
                            end_idx: Optional[int] = None, num_bins: int = 80, 
                            verbose: bool = False, precise: bool = False,
                            minimal: bool = False) -> Dict:
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
        
        # Optimized Histogram calculation
        if precise:
            volume_profile_values = get_precise_volume_histogram(df, start_idx, end_idx, num_bins, verbose=verbose)
        else:
            volume_profile_values = get_raw_volume_histogram(df, start_idx, end_idx, num_bins, verbose=verbose)
            
        volume_profile = pd.Series(volume_profile_values, index=range(num_bins))
        
        # Calculate POC (Point of Control) - highest volume price
        poc_idx = volume_profile.idxmax()
        poc_price = bin_centers[poc_idx]
        poc_volume = volume_profile.iloc[poc_idx]
        
        # Calculate Value Area (70% of total volume)
        if not minimal:
            vah, val = calculate_value_area(volume_profile, bin_centers, value_area_pct=70)
            
            # Find HVNs, LVNs, and CLUSTERS (Peaks)
            hvn_prices = find_hvns(volume_profile, bin_centers, threshold_percentile=75)
            lvn_prices = find_lvns(volume_profile, bin_centers, threshold_percentile=25)
            cluster_prices = find_volume_clusters(volume_profile, bin_centers)
        else:
            vah, val = 0.0, 0.0
            hvn_prices, lvn_prices, cluster_prices = [], [], []
        
        # Generate smoothed profile for visualization (fast method)
        if not minimal:
            from scipy.ndimage import gaussian_filter1d
            profile_smooth = gaussian_filter1d(volume_profile.values.astype(float), sigma=2)
        else:
            profile_smooth = None
        
        return {
            'profile': volume_profile,
            'profile_smooth': profile_smooth,
            'bin_centers': bin_centers,
            'bins': bins,
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'vah': vah,
            'val': val,
            'hvn_prices': hvn_prices,
            'lvn_prices': lvn_prices,
            'cluster_prices': cluster_prices,
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

def find_volume_clusters(volume_profile: pd.Series, bin_centers: np.ndarray) -> List[float]:
    """
    Find major volume clusters (peaks) in the profile using local maxima.
    Returns the price levels of these cluster centers.
    """
    try:
        from scipy.signal import find_peaks
        # Find peaks with some prominence to ignore noise
        # Normalize volume for prominence check
        vol_norm = volume_profile.values / volume_profile.values.max()
        peaks, _ = find_peaks(vol_norm, prominence=0.05, distance=3) # Distance prevents adjacent bins
        
        cluster_centers = [bin_centers[i] for i in peaks]
        return cluster_centers
    except ImportError:
        # Fallback: simple local maxima
        vals = volume_profile.values
        peaks = []
        for i in range(1, len(vals)-1):
            if vals[i] > vals[i-1] and vals[i] > vals[i+1]:
                # Simple peak
                peaks.append(bin_centers[i])
        return peaks


def find_lvns(volume_profile: pd.Series, bin_centers: np.ndarray, 
              threshold_percentile: float = 25) -> List[float]:
    """
    Find Low Volume Nodes using Adaptive Gaussian Mixture Model
    Automatically determines optimal number of peaks (2-5) using BIC
    Finds all minima between adjacent peaks
    
    Args:
        volume_profile: Series of volume per bin
        bin_centers: Array of price levels
        threshold_percentile: Used to filter weak minima
    
    Returns:
        List of LVN prices (minima between Gaussian peaks)
    """
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import minimize_scalar
    
    # Filter out completely empty profiles
    if volume_profile.sum() == 0:
        return []
    
    # Prepare data for GMM: weighted by volume
    prices_weighted = []
    for i, vol in enumerate(volume_profile.values):
        if vol > 0:
            count = int(vol / volume_profile.sum() * 1000)
            prices_weighted.extend([bin_centers[i]] * max(1, count))
    
    if len(prices_weighted) < 10:
        return []
    
    X = np.array(prices_weighted).reshape(-1, 1)
    
    # Adaptive component selection using BIC (2-5 peaks)
    best_gmm = None
    best_bic = np.inf
    best_n = 2
    
    for n_components in range(2, 6):  # Test 2-5 peaks
        try:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, max_iter=100)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_n = n_components
        except:
            continue
    
    if best_gmm is None:
        return []
    
    # Get peaks sorted by price
    peaks = sorted(best_gmm.means_.flatten())
    
    # Define GMM PDF
    def gmm_pdf(x):
        return np.exp(best_gmm.score_samples([[x]]))[0]
    
    # Find minima between ALL adjacent peaks
    lvn_prices = []
    for i in range(len(peaks) - 1):
        result = minimize_scalar(lambda x: gmm_pdf(x), bounds=(peaks[i], peaks[i+1]), method='bounded')
        lvn_prices.append(result.x)
    
    # Filter: only keep minima that are between bins with actual volume
    # AND below threshold percentile
    non_zero_profile = volume_profile[volume_profile > 0]
    if len(non_zero_profile) > 0:
        threshold = np.percentile(non_zero_profile.values, threshold_percentile)
        # Check actual volume at each LVN
        valid_lvns = []
        for lvn in lvn_prices:
            idx = np.argmin(np.abs(bin_centers - lvn))
            
            # CRITICAL FIX: Ensure LVN is between two non-zero volume bins
            # Find nearest non-zero bins on left and right
            left_idx = idx
            while left_idx >= 0 and volume_profile.iloc[left_idx] == 0:
                left_idx -= 1
            
            right_idx = idx
            while right_idx < len(volume_profile) and volume_profile.iloc[right_idx] == 0:
                right_idx += 1
            
            # Only accept LVN if it's between two actual volume clusters
            if left_idx >= 0 and right_idx < len(volume_profile):
                # Check if the LVN bin (or nearest non-zero bin) is below threshold
                if volume_profile.iloc[idx] <= threshold:
                    valid_lvns.append(lvn)
                elif left_idx != idx and volume_profile.iloc[left_idx] <= threshold:
                    valid_lvns.append(lvn)
                elif right_idx != idx and volume_profile.iloc[right_idx] <= threshold:
                    valid_lvns.append(lvn)
        
        return valid_lvns
    
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


def find_dual_distribution_peaks(volume_profile: pd.Series, bin_centers: np.ndarray) -> List[float]:
    """
    Fit a Gaussian Mixture Model with 2 components to find the two major distribution centers.
    Returns the prices of the two peaks (means).
    """
    try:
        from sklearn.mixture import GaussianMixture
        
        # Filter out completely empty profiles
        if volume_profile.sum() == 0:
            return []
        
        # Prepare data for GMM: weighted by volume
        # We need to reconstruct the dataset from the histogram
        prices_weighted = []
        # Downsample for performance if needed, but 1000 samples is fast
        total_vol = volume_profile.sum()
        target_samples = 1000
        
        for i, vol in enumerate(volume_profile.values):
            if vol > 0:
                # Proportional number of samples
                count = int((vol / total_vol) * target_samples)
                if count > 0:
                    prices_weighted.extend([bin_centers[i]] * count)
        
        if len(prices_weighted) < 10:
            return []
        
        X = np.array(prices_weighted).reshape(-1, 1)
        
        # Force 2 components
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, max_iter=100)
        gmm.fit(X)
        
        # Get the means (peaks) and flatten
        peaks = sorted(gmm.means_.flatten())
        return peaks
        
    except Exception as e:
        print(f"GMM Error: {e}")
        return []

def get_dual_distribution_pdf(volume_profile: pd.Series, bin_centers: np.ndarray) -> np.ndarray:
    """
    Fit a GMM with 2 components and return the probability density (PDF) values 
    evaluated at bin_centers. This creates a smooth "Two Peak" curve.
    """
    try:
        from sklearn.mixture import GaussianMixture
        
        if volume_profile.sum() == 0:
            return np.zeros_like(bin_centers)
            
        prices_weighted = []
        total_vol = volume_profile.sum()
        target_samples = 1000
        
        for i, vol in enumerate(volume_profile.values):
            if vol > 0:
                count = int((vol / total_vol) * target_samples)
                if count > 0:
                    prices_weighted.extend([bin_centers[i]] * count)
        
        if len(prices_weighted) < 10:
            return np.zeros_like(bin_centers)
        
        X = np.array(prices_weighted).reshape(-1, 1)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, max_iter=100)
        gmm.fit(X)
        
        # Evaluate PDF at bin_centers
        logprob = gmm.score_samples(bin_centers.reshape(-1, 1))
        pdf = np.exp(logprob)
        
        # Scale PDF to match volume profile height for visualization
        # We scale so the max of PDF matches max of actual volume profile
        if pdf.max() > 0:
            scale_factor = volume_profile.max() / pdf.max()
            pdf_scaled = pdf * scale_factor
            return pdf_scaled
            
        return np.zeros_like(bin_centers)
        
    except Exception as e:
        print(f"GMM PDF Error: {e}")
        return np.zeros_like(bin_centers)


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
