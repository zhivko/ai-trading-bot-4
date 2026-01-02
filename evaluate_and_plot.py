import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
from train_w_pattern_vp import PatternDetectorCNN, prepare_data  # Import from our training script

def evaluate_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")
    
    # 1. Load Data
    PATH = "BTCUSDT_1h_data.csv"
    if not os.path.exists(PATH):
        print("Data file not found.")
        return
        
    print("Loading data...")
    X_ts, X_vp, y = prepare_data(PATH)
    
    # 2. Load Model
    model = PatternDetectorCNN(window_size=100, vp_bins=80).to(device)
    model.load_state_dict(torch.load("stoch_vp_detector_5090.pth"))
    model.eval()
    
    # 3. Inference
    print("Running inference...")
    batch_size = 1024
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            b_ts = torch.FloatTensor(X_ts[i:i+batch_size]).to(device)
            b_vp = torch.FloatTensor(X_vp[i:i+batch_size]).to(device)
            
            # Forward
            out = model(b_ts, b_vp)
            prob = torch.sigmoid(out).cpu().numpy()
            preds.extend(prob)
            
    preds = np.array(preds).flatten()
    
    # 4. Plotting
    print("Generating plots...")
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # Subplot 2: Precision-Recall Curve (Better for imbalanced data)
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y, preds)
    
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_plot.png')
    print("Saved performance_plot.png")
    
    # Print Metrics
    binary_preds = (preds > 0.5).astype(int)
    cm = confusion_matrix(y, binary_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
import os
if __name__ == "__main__":
    evaluate_and_plot()
