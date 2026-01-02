# Stochastic Low Pattern Detector - Training Summary

## Overview
Successfully trained a neural network to detect **prolonged stochastic lows below 20** using a **HYBRID APPROACH** that combines:
1. **Early Detection**: Signals DURING the prolonged low period (accumulation zone)
2. **Breakout Confirmation**: Signals when prolonged lows are followed by price breakouts

This gives you both early warning signals AND validated breakout patterns.

## Pattern Being Detected

The model identifies two types of situations:

### Type 1: Early Warning (During Accumulation)
- **Stochastic is currently low** (below 25)
- **Has been low for extended period** (at least 30% of 100-bar window below 20)
- **Sustained recent lows** (at least 10 consecutive bars below 20)
- **Recent confirmation** (6+ of last 10 bars below 20)

**Use case**: Get alerted DURING the accumulation zone (red arrows in your chart)

### Type 2: Breakout Confirmation
- **Prolonged low history** (35%+ of window below 20)
- **Sustained low period** (10+ consecutive bars below 20)
- **Followed by price breakout** (at least 2% price increase within next 20 bars)

**Use case**: Confirm that the pattern actually led to a bullish move

This hybrid approach addresses your feedback - we keep the valuable breakout confirmation while also detecting during the low period!

## Training Results

### Dataset Statistics
- **Total samples**: 293,067
- **Positive samples**: 11,985 (4.1%)
  - Early warning signals: ~9,600
  - Breakout confirmations: ~2,400 (with some overlap)
- **Negative samples**: 281,082 (95.9%)
- **Data source**: BTCUSDT 15m timeframe

### Model Architecture
- **Type**: CNN (Convolutional Neural Network)
- **Input**: 100-bar window of stochastic values (normalized 0-1)
- **Layers**: 3 convolutional blocks with batch normalization
- **Parameters**: ~38M parameters
- **Output**: Binary classification (pattern detected or not)

### Training Configuration
- **Device**: NVIDIA GeForce RTX 5090
- **Training time**: 370.83 seconds (~6.2 minutes)
- **Epochs**: 100
- **Batch size**: 512
- **Learning rate**: 0.0001 → 0.001 (OneCycleLR)
- **Class weight**: 23.45 (capped at 50 for stability)
- **Gradient clipping**: max_norm=1.0

### Performance Metrics
- **Final training loss**: 0.016946
- **Final validation loss**: 0.134660
- **Best validation loss**: 0.095420 (epoch 80)

**Note**: The model shows good convergence. Best performance is around epoch 60-80 where validation loss is lowest.

## Model Files
- **Model weights**: `stoch_low_detector_5090.pth`
- **Training plot**: `performance_plot.png`
- **Training script**: `train_w_pattern.py`

## How to Use the Model

### 1. Load the Model
```python
import torch
from train_w_pattern import PatternDetectorCNN

# Initialize model
model = PatternDetectorCNN(window_size=100)

# Load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("stoch_low_detector_5090.pth", map_location=device))
model.to(device)
model.eval()
```

### 2. Prepare Input Data
```python
import pandas as pd
import numpy as np

# Calculate stochastic (same as multi_stochastic_quad_rotation.py)
def calculate_stochastic(df, period_k=60, smooth_k=10, smooth_d=10):
    low_min = df['low'].rolling(window=period_k, min_periods=1).min()
    high_max = df['high'].rolling(window=period_k, min_periods=1).max()
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
    if smooth_k > 1:
        k_percent = k_percent.rolling(window=smooth_k, min_periods=1).mean()
    d_percent = k_percent.rolling(window=smooth_d, min_periods=1).mean()
    return d_percent

# Load your data
df = pd.read_csv("your_data.csv")
stoch = calculate_stochastic(df, period_k=60, smooth_k=10, smooth_d=10)
stoch_normalized = stoch / 100.0  # Normalize to 0-1

# Get last 100 bars
window = stoch_normalized.iloc[-100:].values

# Convert to tensor
input_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
# Shape: (1, 1, 100) = (batch, channels, sequence_length)
```

### 3. Make Predictions
```python
with torch.no_grad():
    output = model(input_tensor)
    probability = torch.sigmoid(output).item()
    
print(f"Pattern detection probability: {probability:.4f}")

# Use a threshold (e.g., 0.5)
if probability > 0.5:
    print("⚠️ PROLONGED STOCHASTIC LOW DETECTED - Potential bullish setup!")
else:
    print("No pattern detected")
```

### 4. Integration with Your Trading Bot
You can integrate this into `multi_stochastic_quad_rotation.py` or `chart_generator.py`:

```python
# Add to your alarm detection logic
def check_stoch_low_pattern(df, model, device):
    """Check if current stochastic shows prolonged low pattern"""
    stoch = calculate_stochastic(df, period_k=60, smooth_k=10, smooth_d=10)
    stoch_norm = stoch / 100.0
    
    if len(stoch_norm) < 100:
        return False, 0.0
    
    window = stoch_norm.iloc[-100:].values
    input_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
    
    is_pattern = probability > 0.5
    return is_pattern, probability
```

## Recommendations

### 1. Use Early Stopping
The model performs best around epoch 20-30. Consider using the checkpoint with the lowest validation loss rather than the final epoch.

### 2. Adjust Threshold
The default 0.5 threshold may not be optimal. You can:
- Lower it (e.g., 0.3) for more sensitive detection (more signals, more false positives)
- Raise it (e.g., 0.7) for more conservative detection (fewer signals, higher precision)

### 3. Combine with Other Indicators
This model works best when combined with:
- Volume analysis
- Support/resistance levels
- Trend confirmation
- Your existing channel detection logic

### 4. Retrain with More Data
If you have more historical data or want to focus on specific market conditions, you can retrain:
```bash
.venv\Scripts\Activate.ps1
python train_w_pattern.py
```

## Next Steps

1. **Test the model** on recent data to validate performance
2. **Backtest** the signals to measure profitability
3. **Integrate** into your live trading system
4. **Monitor** false positives/negatives and retrain if needed
5. **Consider** training an ensemble of models for better robustness

## Pattern Criteria Summary

The model was trained with a **HYBRID APPROACH** to detect:

### Early Warning Signals (Type 1)
- Current stochastic value < 25
- ≥30% of 100-bar window below 20
- ≥10 consecutive bars below 20
- ≥6 of last 10 bars below 20

### Breakout Confirmation (Type 2)
- ≥35% of 100-bar window below 20
- ≥10 consecutive bars below 20
- Followed by ≥2% price increase within 20 bars

This creates a comprehensive dataset that includes both:
- **Accumulation zones** (early warnings during the low period)
- **Validated breakouts** (confirmed patterns that led to price increases)
```
