import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class CLIPBasedModel(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=False):
        super().__init__()

        try:
            self.vision_backbone = CLIPVisionModel.from_pretrained(model_name)
        except Exception as e:
            print(f"CLIPVisionModel ({model_name}) load failed: {e}")
            raise

        if freeze_backbone:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            print("CLIP backbone parameters frozen.")

        self.feature_dim = self.vision_backbone.config.hidden_size
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, pixel_values):
        outputs = self.vision_backbone(pixel_values = pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.head(pooled_output)
        return logits
