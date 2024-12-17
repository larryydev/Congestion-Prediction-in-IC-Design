import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from data_loader import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

class CongestionDataset(Dataset):
    def __init__(self, target_size=(256, 256)) -> None:
        self.target_size = target_size
        self.all_files, self.all_files_size = DataLoader.get_all_file_paths()
        # self.cell_density_files = self.all_files['cell_density']
        # self.RUDY_long_files = self.all_files['RUDY_long']
        # self.RUDY_pin_long_files = self.all_files['RUDY_pin_long']
        self.RUDY_files = self.all_files['RUDY']
        # self.RUDY_short_files = self.all_files['RUDY_short']
        self.RUDY_pin_files = self.all_files['RUDY_pin']
        self.macro_region = self.all_files['macro_region']
        # self.congestion_eGR_vertical_overflow_files = self.all_files['congestion_eGR_vertical_overflow']
        # self.congestion_eGR_horizontal_overflow_files = self.all_files['congestion_eGR_horizontal_overflow']
        # self.congestion_eGR_vertical_util_files = self.all_files['congestion_eGR_vertical_util']
        # self.congestion_eGR_horizontal_util_files = self.all_files['congestion_eGR_horizontal_util']
        self.congestion_GR_vertical_overflow_files = self.all_files['congestion_GR_vertical_overflow']
        self.congestion_GR_horizontal_overflow_files = self.all_files['congestion_GR_horizontal_overflow']
        # self.congestion_GR_vertical_util_files = self.all_files['congestion_GR_vertical_util']
        # self.congestion_GR_horizontal_util_files = self.all_files['congestion_GR_horizontal_util']

    def load_and_resize(self, path):
        data = np.load(path)
        resized_data = cv2.resize(data, self.target_size)
        tensor_data = torch.from_numpy(resized_data).unsqueeze(0).float()
        return tensor_data

    def __len__(self):
        return self.all_files_size

    def __getitem__(self, index):
        input_data = {
            # 'cell_density': self.load_and_resize(self.cell_density_files[index]),
            # 'RUDY_long': self.load_and_resize(self.RUDY_long_files[index]),
            # 'RUDY_pin_long': self.load_and_resize(self.RUDY_pin_long_files[index]),
            'RUDY': self.load_and_resize(self.RUDY_files[index]),
            # 'RUDY_short': self.load_and_resize(self.RUDY_short_files[index]),
            'RUDY_pin': self.load_and_resize(self.RUDY_pin_files[index]),
            'macro_region': self.load_and_resize(self.macro_region[index])
        }
        
        target_data = {
            # 'eGR_vertical_overflow': self.load_and_resize(self.congestion_eGR_vertical_overflow_files[index]),
            # 'eGR_horizontal_overflow': self.load_and_resize(self.congestion_eGR_horizontal_overflow_files[index]),
            # 'eGR_vertical_util': self.load_and_resize(self.congestion_eGR_vertical_util_files[index]),
            # 'eGR_horizontal_util': self.load_and_resize(self.congestion_eGR_horizontal_util_files[index]),
            'GR_vertical_overflow': self.load_and_resize(self.congestion_GR_vertical_overflow_files[index]),
            'GR_horizontal_overflow': self.load_and_resize(self.congestion_GR_horizontal_overflow_files[index]),
            # 'GR_vertical_util': self.load_and_resize(self.congestion_GR_vertical_util_files[index]),
            # 'GR_horizontal_util': self.load_and_resize(self.congestion_GR_horizontal_util_files[index])
        }

        input_data = {k: v for k, v in input_data.items() if v is not None}
        target_data = {k: v for k, v in target_data.items() if v is not None}

        input_tensor = torch.cat(list(input_data.values()), dim=0)
        target_tensor = torch.cat(list(target_data.values()), dim=0)
        
        return input_tensor, target_tensor

def visualize_samples(dataset, num_samples=3):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(num_samples):
        input_data, target_data = dataset[i]
        
        if input_data.shape[0] == 1:
            axes[0, i].imshow(input_data[0].numpy(), cmap='viridis')
        else:
            axes[0, i].imshow(input_data[0].numpy(), cmap='viridis')
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        
        if target_data.shape[0] == 1:
            axes[1, i].imshow(target_data[0].numpy(), cmap='viridis')
        else:
            axes[1, i].imshow(target_data[0].numpy(), cmap='viridis')
        axes[1, i].set_title(f'Target {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    dataset = CongestionDataset()

    visualize_samples(dataset)