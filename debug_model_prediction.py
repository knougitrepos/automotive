import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from pathlib import Path

# Config hardcoded for simplicity matching model_config.yaml
HIDDEN_DIMS = [512, 256]
DROPOUT = 0.3

class DrivingPolicyModel(nn.Module):
    def __init__(self, encoder_type='resnet50'):
        super().__init__()
        
        if encoder_type == 'resnet50':
            self.encoder = models.resnet50(pretrained=False)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            self.encoder = models.resnet34(pretrained=False)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.feature_dim, HIDDEN_DIMS[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[1], 3)
        )
        
        self.steer_activation = nn.Tanh()
        self.throttle_brake_activation = nn.Sigmoid()
    
    def forward(self, x):
        features = self.encoder(x)
        raw_output = self.policy_head(features)
        
        steer = self.steer_activation(raw_output[:, 0:1])
        throttle = self.throttle_brake_activation(raw_output[:, 1:2])
        brake = self.throttle_brake_activation(raw_output[:, 2:3])
        
        return torch.cat([steer, throttle, brake], dim=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DrivingPolicyModel(encoder_type='resnet50').to(device)
    
    checkpoint_path = Path('c:/git/automotive/checkpoints/bc/best_bc_model.pth')
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
            if 'val_loss' in checkpoint:
                print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
            if 'epoch' in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    model.eval()
    
    print("\nModel Architecture loaded. Running inference checks...")
    
    # 1. Zero Input
    dummy_input = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print("\n[Input: Zeros]")
    print(f"Steer:    {output[0][0].item():.4f} (Expected ~0.0)")
    print(f"Throttle: {output[0][1].item():.4f}")
    print(f"Brake:    {output[0][2].item():.4f}")
    
    # 2. Random Input
    rand_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output_rand = model(rand_input)
        
    print("\n[Input: Random Noise]")
    print(f"Steer:    {output_rand[0][0].item():.4f}")
    print(f"Throttle: {output_rand[0][1].item():.4f}")
    print(f"Brake:    {output_rand[0][2].item():.4f}")

    if output[0][2].item() > 0.1:
         print("\nWARNING: High brake value on zero input. Car might be stuck braking.")

if __name__ == "__main__":
    main()
