import torch
import matplotlib.pyplot as plt
import numpy as np
from congestion_dataset import CongestionDataset
from model import CongestionModel

def load_trained_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def visualize_predictions(model, dataset, num_samples=3, device='cpu'):
    model = model.to(device)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for idx in range(num_samples):
        inputs, targets = dataset[idx]
        inputs = inputs.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(inputs)
        
        inputs = inputs.cpu().squeeze(0)
        targets = targets.cpu()
        predictions = predictions.cpu().squeeze(0)
        
        axes[idx, 0].imshow(inputs[0].numpy(), cmap='viridis')
        axes[idx, 0].set_title(f'Input\n')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(targets[0].numpy(), cmap='viridis')
        axes[idx, 1].set_title(f'Target\n')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(predictions[0].numpy(), cmap='viridis')
        axes[idx, 2].set_title(f'Prediction\n')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    dataset = CongestionDataset()
    input_data, target_data = dataset[0]
    model = CongestionModel(dim_in=input_data.shape[0], dim_out=target_data.shape[0])
    
    checkpoint_path = 'checkpoint/model3/best_model.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = load_trained_model(checkpoint_path, model)
    
    visualize_predictions(model, dataset, num_samples=10, device=device)

if __name__ == '__main__':
    main()