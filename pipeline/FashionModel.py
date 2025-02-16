import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B1_Weights # Import weights

class FashionModel(nn.Module):
    def __init__(self, num_brand_features, num_classes):
        super(FashionModel, self).__init__()

        # Load EfficientNet from torchvision
        self.efficientnet = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        
        # Remove the classifier head (fc layer)
        self.efficientnet.classifier = nn.Identity()

        # Brand feature processing
        self.brand_fc = nn.Sequential(
            nn.Linear(num_brand_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Combined FC layer
        self.combined_fc = nn.Sequential(
            nn.Linear(1280 + 64, 512),  # 1280 from EfficientNet-b0 output
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, brand):
        image_features = self.efficientnet(image)  # Extract image features
        brand_features = self.brand_fc(brand)  # Process brand features

        combined_features = torch.cat((image_features, brand_features), dim=1)  # Concatenate features
        output = self.combined_fc(combined_features)  # Final prediction
        return output