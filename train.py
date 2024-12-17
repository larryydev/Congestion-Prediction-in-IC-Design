import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
from tqdm import tqdm
import os
from pathlib import Path

from congestion_dataset import CongestionDataset
from model import CongestionModel

EPOCHS = 10
BATCH_SIZE = 16  # Reduced batch size to help with memory
LEARNING_RATE = 0.001
DEVICE = ""
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

CHECKPOINT_DIR = Path("./checkpoint/model3")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("./log")
LOG_DIR.mkdir(exist_ok=True)

class CongestionLoss(nn.Module):
    def __init__(self, ssim_weight=0.5):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        self.ssim_weight = ssim_weight
        
    def forward(self, pred, target):
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)
        
        # SSIM Loss (1 - SSIM because we want to minimize)
        ssim_loss = 1 - self.ssim(pred, target)
        
        # Combine losses
        total_loss = (1 - self.ssim_weight) * mse_loss + self.ssim_weight * ssim_loss
        
        return total_loss

def accuracy_fn(y_true, y_pred):
    y_true = y_true.to(y_pred.device)
    mse = F.mse_loss(y_true, y_pred)
    return mse

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filename)

def train(model, train_loader, val_loader):
    model = model.to(DEVICE)
    loss_fn = CongestionLoss(ssim_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min")
    
    best_val_loss = float('inf')

    # TRAINING
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        
        # Create progress bar for training
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
        for inputs, targets in train_progress:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            metrics = accuracy_fn(y_true=targets, y_pred=outputs)
            
            train_loss += loss.item()
            train_acc += metrics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear cache if using MPS
            if DEVICE == "mps":
                torch.mps.empty_cache()
            
            # Update progress bar
            train_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{metrics:.4f}'
            })

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0 
        val_acc = 0

        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Valid]')
        
        with torch.no_grad():
            for inputs, targets in val_progress:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)

                loss = loss_fn(outputs, targets)
                metrics = accuracy_fn(y_true=targets, y_pred=outputs)
                
                val_loss += loss.item()
                val_acc += metrics
                
                # Clear cache if using MPS
                if DEVICE == "mps":
                    torch.mps.empty_cache()
                
                val_progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'accuracy': f'{metrics:.4f}'
                })

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        scheduler.step(val_loss)

        # Save checkpoint
        checkpoint_path = CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pt'
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = CHECKPOINT_DIR / 'best_model.pt'
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_model_path)

        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Train Accu: {train_acc:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        print(f'Val Accu: {val_acc:.6f}')
        print('----------------------------------------\n')


if __name__ == '__main__':
    torch.manual_seed(42)

    print(f'Training on: {DEVICE}')

    dataset = CongestionDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                            [train_size, val_size])
    
    train_loader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=0)
    
    val_loader = DataLoader(val_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=0)
    
    input, target = train_dataset[0]

    model = CongestionModel(dim_in=input.shape[0],
                          dim_out=target.shape[0])
    
    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader)