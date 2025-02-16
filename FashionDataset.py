import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class FashionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.label_columns = [
            col for col in self.data.columns
            if col != "image_path" and not col.startswith("brand_")
        ]

        self.brand_columns = [
            col for col in self.data.columns
            if col.startswith("brand_")
        ]

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(self.data.iloc[idx][self.label_columns].astype(float).values, dtype=torch.float32)

        brand = torch.tensor(self.data.iloc[idx][self.brand_columns].astype(float).values, dtype=torch.float32)

        return image, brand, labels
    


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
])