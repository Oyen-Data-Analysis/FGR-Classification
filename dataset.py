# You need IMAGES OF SEGMENTED PLACENTAS to run this script

import cv2
from torch.utils.data import Dataset

class PlacentaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        return image, label