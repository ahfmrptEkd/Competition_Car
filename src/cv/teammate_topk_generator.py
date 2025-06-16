import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data.enhanced_dataset import EnhancedImageDataset
from src.config import DEVICE, DATA_ROOT_DIR, TRAIN_IMG_DIR

# 팀원 모델 구조만 별도 정의
class TeammateConvNeXt(nn.Module):
    """팀원과 동일한 ConvNeXt 구조"""
    def __init__(self, num_classes):
        super(TeammateConvNeXt, self).__init__()
        # 팀원과 완전 동일한 구조
        self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)       
        x = self.head(x) 
        return x

def generate_train_topk(model, output_path, top_k=5, batch_size=32):
    """학습 데이터 Top-K 생성 (기존 Dataset 활용)"""
    
    # 기존 Dataset 사용 (torchvision 호환 processor)
    processor = {
        "size": {"height": 224, "width": 224},
        "do_center_crop": True,
        "crop_size": {"height": 224, "width": 224},
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }
    
    train_dataset = EnhancedImageDataset(
        data_dir=TRAIN_IMG_DIR,
        processor=processor,
        mode='train',
        model_type='timm'  # timm 스타일 전처리 사용
    )
    
    class_names = train_dataset.classes
    print(f"✅ 클래스 수: {len(class_names)}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=lambda batch: [item for item in batch if item is not None]
    )
    
    model.eval()
    all_predictions = {}
    
    print(f"🎯 학습 데이터 Top-{top_k} 생성 중...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing train batches")):
            if not batch:
                continue
                
            # 배치 데이터 준비
            pixel_values = torch.stack([item['pixel_values'] for item in batch]).to(DEVICE)
            labels = torch.stack([item['label'] for item in batch])
            
            outputs = model(pixel_values)
            probs = F.softmax(outputs, dim=1)
            
            for i in range(len(batch)):
                img_info = train_dataset.get_img_info(batch_idx * batch_size + i)
                image_id = os.path.splitext(img_info['image_id'])[0]  # 확장자 제거
                
                # Top-K 추출
                img_probs = probs[i].cpu().numpy()
                top_k_indices = np.argsort(img_probs)[-top_k:][::-1]
                
                top_k_predictions = []
                for k_idx in top_k_indices:
                    class_name = class_names[k_idx]
                    confidence = float(img_probs[k_idx])
                    top_k_predictions.append({
                        "class_name": class_name,
                        "confidence": round(confidence, 6)
                    })
                
                all_predictions[image_id] = top_k_predictions
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 학습 Top-{top_k} 저장: {output_path}")
    print(f"📊 총 {len(all_predictions)}개 이미지 처리")
    return output_path

def generate_test_topk(model, output_path, top_k=5, batch_size=32):
    """테스트 데이터 Top-K 생성 (기존 Dataset 활용)"""
    
    processor = {
        "size": {"height": 224, "width": 224},
        "do_center_crop": True,
        "crop_size": {"height": 224, "width": 224},
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }
    
    test_dataset = EnhancedImageDataset(
        data_dir=DATA_ROOT_DIR,
        processor=processor,
        mode='test',
        test_csv_path=os.path.join(DATA_ROOT_DIR, 'test.csv'),
        model_type='timm'
    )
    
    sample_submission = pd.read_csv(os.path.join(DATA_ROOT_DIR, 'sample_submission.csv'))
    class_names = sample_submission.columns[1:].tolist()
    
    # DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=lambda batch: [item for item in batch if item is not None]
    )
    
    model.eval()
    all_predictions = {}
    
    print(f"🎯 테스트 데이터 Top-{top_k} 생성 중...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing test batches")):
            if not batch:
                continue
                
            pixel_values = torch.stack([item['pixel_values'] for item in batch]).to(DEVICE)
            img_ids = [item['img_id'] for item in batch]
            
            outputs = model(pixel_values)
            probs = F.softmax(outputs, dim=1)
            
            for i, img_id in enumerate(img_ids):
                img_probs = probs[i].cpu().numpy()
                top_k_indices = np.argsort(img_probs)[-top_k:][::-1]
                
                top_k_predictions = []
                for k_idx in top_k_indices:
                    class_name = class_names[k_idx]
                    confidence = float(img_probs[k_idx])
                    top_k_predictions.append({
                        "class_name": class_name,
                        "confidence": round(confidence, 6)
                    })
                
                all_predictions[img_id] = top_k_predictions
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 테스트 Top-{top_k} 저장: {output_path}")
    print(f"📊 총 {len(all_predictions)}개 이미지 처리")
    return output_path

def main():
    MODEL_PATH = "models/choi/timm_convnext_after.pth"
    OUTPUT_DIR = "outputs"
    TOP_K = 5
    NUM_CLASSES = 396
    
    print("🚀 팀원 모델 Top-K 생성 (기존 Dataset 활용)")
    
    print("📦 팀원 모델 로딩...")
    model = TeammateConvNeXt(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.to(DEVICE)
    print("✅ 모델 로드 완료")
    
    print("\n🎯 Top-K 예측 생성...")
    
    train_output = os.path.join(OUTPUT_DIR, "train_teammate_convnext_top_5.json")
    generate_train_topk(model, train_output, TOP_K)
    
    test_output = os.path.join(OUTPUT_DIR, "test_teammate_convnext_top_5.json")
    generate_test_topk(model, test_output, TOP_K)
    
    print("\n🎉 완료! 이제 기존 VLM 파이프라인에서 이 JSON들을 사용하세요:")
    print(f"📁 {train_output}")
    print(f"📁 {test_output}")

if __name__ == "__main__":
    main()