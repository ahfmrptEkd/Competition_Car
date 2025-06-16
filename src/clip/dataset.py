import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, data_dir,
                 processor,
                 mode = 'train',
                 augment_transform = None,
                 class_names = None,
                 class_to_idx = None,
                 test_csv_path = None
                 ):
        self.data_dir = data_dir
        self.processor = processor
        self.mode = mode
        self.augment_transform = augment_transform
        self.samples = []
        self.img_ids = []

        if mode in ['train', 'val']:
            self.classes = class_names if class_names is not None else sorted(os.listdir(data_dir))
            self.class_to_idx = class_to_idx if class_to_idx is not None else {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            for cls_name in self.classes:
                cls_folder = os.path.join(data_dir, cls_name)

                if not os.path.isdir(cls_folder): continue

                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

        elif mode == 'test':
            if test_csv_path is None:
                raise ValueError("test_csv_path is not defined")
            
            test_df = pd.read_csv(test_csv_path)

            for _, row in test_df.iterrows():
                if row['img_path'].startswith('./'):
                    img_path = row['img_path'][2:]
                else:
                    img_path = row['img_path']
                
                img_full_path = os.path.join(data_dir, img_path)
                
                self.samples.append(img_full_path)
                self.img_ids.append(row['ID'])
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            img_path, label = self.samples[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                print(f"Warning: {self.mode} image not found: {img_path}. Skipping.")
                return None

            if self.augment_transform:
                image = self.augment_transform(image)

            processed_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = processed_inputs['pixel_values'].squeeze(0)
            return {"pixel_values": pixel_values, "label": torch.tensor(label, dtype=torch.long)}

        elif self.mode == 'test':
            img_path = self.samples[idx]
            img_id = self.img_ids[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                print(f"Warning: Test image not found: {img_path}. Skipping.")
                return None

            if self.augment_transform:
                 image = self.augment_transform(image)


            processed_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = processed_inputs['pixel_values'].squeeze(0)
            return {"pixel_values": pixel_values, "img_id": img_id}
            
    def get_img_info(self, idx):
        if self.mode in ['train', 'val']:
            img_path, label_idx = self.samples[idx]
            image_id = os.path.basename(img_path)
            abs_path = os.path.abspath(img_path)
            label_text = self.idx_to_class[label_idx]
            return {"image_id": image_id, "abs_path": abs_path, "label_text": label_text}
        elif self.mode == 'test':
            img_path = self.samples[idx]
            img_id = self.img_ids[idx]
            abs_path = os.path.abspath(img_path)
            return {"image_id": img_id, "abs_path": abs_path, "label_text": None} 
        else:
            raise ValueError(f"get_img_info not supported for mode: {self.mode}")
            