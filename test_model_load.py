import torch
import torch.nn as nn
import os

# Copy the model class from train_w_pattern.py
class PatternDetectorCNN(nn.Module):
    def __init__(self, window_size=100):
        super(PatternDetectorCNN, self).__init__()
        # Input channels = 1 (Stochastic)
        self.features = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Test loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PatternDetectorCNN()
model_path = "stoch_low_detector_5090.pth"

if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        print(f"Model file: {model_path}")
        print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found: {model_path}")
