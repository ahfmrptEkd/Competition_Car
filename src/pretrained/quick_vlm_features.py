import torch
import json
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os
import pandas as pd
import argparse
from config.config import TRAIN_IMG_DIR, DATA_ROOT_DIR
from utils.data.enhanced_dataset import EnhancedImageDataset

class CLIPBasedFeatureExtractor:
    def __init__(self):
        # CLIP 사용 (안정적이고 빠름)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ CLIP model loaded on {self.device}")
    
    def extract_features(self, image_path, model_a_predictions):
        """이미지와 모델 A 예측으로부터 특징 추출"""
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            
            # 모델 A 예측을 텍스트로 변환
            pred_texts = [
                f"This is a {pred['class_name']} car"
                for pred in model_a_predictions[:3]
            ]
            
            # 기본 설명도 추가
            pred_texts.append("This is a car image")
            
            # CLIP으로 이미지 특징 추출
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # 텍스트 특징 추출
            text_inputs = self.processor(text=pred_texts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                # 이미지 특징
                image_features = self.model.get_image_features(**image_inputs)
                
                # 텍스트 특징 (여러 텍스트의 평균)
                text_features = self.model.get_text_features(**text_inputs)
                text_features_mean = text_features.mean(dim=0, keepdim=True)
                
                # 이미지와 텍스트 특징 결합
                combined_features = torch.cat([image_features, text_features_mean], dim=1)
                
            return combined_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # 오류 시 더미 특징 반환
            return np.zeros(512 + 512, dtype=np.float32)  # CLIP feature size

def quick_clip_pipeline(train_images, train_labels, train_predictions, 
                       test_images, test_predictions, args=None):
    """빠른 CLIP 기반 파이프라인"""
    
    extractor = CLIPBasedFeatureExtractor()
    
    print("Extracting features from train images...")
    train_features = []
    train_ids = []
    labels = []
    
    # 학습 데이터 처리
    matched_train_count = 0
    for pred_id in tqdm(list(train_predictions.keys())[:], desc="Processing train images"):
        img_id_with_ext = pred_id + ".jpg"
        
        if img_id_with_ext in train_images and img_id_with_ext in train_labels:
            img_path = train_images[img_id_with_ext]
            if os.path.exists(img_path):
                features = extractor.extract_features(img_path, train_predictions[pred_id])
                train_features.append(features)
                train_ids.append(pred_id)
                labels.append(train_labels[img_id_with_ext])
                matched_train_count += 1
    
    print(f"✅ Processed {matched_train_count} train images")
    
    print("Extracting features from test images...")  
    test_features = []
    test_ids = []
    
    # 테스트 데이터 처리
    matched_test_count = 0
    for pred_id in tqdm(list(test_predictions.keys())[:], desc="Processing test images"):
        if pred_id in test_images:
            img_path = test_images[pred_id]
            if os.path.exists(img_path):
                features = extractor.extract_features(img_path, test_predictions[pred_id])
                test_features.append(features)
                test_ids.append(pred_id)
                matched_test_count += 1
    
    print(f"✅ Processed {matched_test_count} test images")
    
    if not train_features or not test_features:
        print("❌ No features extracted!")
        return
    
    # 저장 경로 결정
    if args and getattr(args, 'train_output', None) and getattr(args, 'test_output', None):
        train_output = args.train_output
        test_output = args.test_output
    else:
        train_output = 'outputs/quick_vlm_train_features.npz'
        test_output = 'outputs/quick_vlm_test_features.npz'
    
    # 저장
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(test_output), exist_ok=True)
    
    np.savez(train_output,
             features=np.vstack(train_features),
             image_ids=train_ids,
             labels=labels)
             
    np.savez(test_output,
             features=np.vstack(test_features), 
             image_ids=test_ids)
    
    print("✅ Quick CLIP features extracted!")
    print(f"Train features: {len(train_features)} samples, shape: {np.vstack(train_features).shape}")
    print(f"Test features: {len(test_features)} samples, shape: {np.vstack(test_features).shape}")
    print(f"Saved to:")
    print(f"  Train: {train_output}")
    print(f"  Test: {test_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract VLM features using CLIP")
    
    parser.add_argument('--train_json', type=str, 
                       help='Path to train top-k JSON file')
    parser.add_argument('--test_json', type=str,
                       help='Path to test top-k JSON file')
    parser.add_argument('--train_output', type=str,
                       help='Path to save train features')
    parser.add_argument('--test_output', type=str,
                       help='Path to save test features')
    
    args = parser.parse_args()
    
    # argparse가 제공되면 그것을 사용, 아니면 기본값 사용 (하위 호환성)
    if args.train_json and args.test_json:
        # Command line arguments 사용
        train_json_path = args.train_json
        test_json_path = args.test_json
    else:
        # 기본값 사용 (기존 코드와 호환)
        train_json_path = 'outputs/train_timm_efficientnet_b4_top_5.json'
        test_json_path = 'outputs/test_timm_efficientnet_b4_top_5.json'
    
    print(f"Loading predictions from:")
    print(f"  Train: {train_json_path}")
    print(f"  Test: {test_json_path}")
    
    # JSON 파일들 로드
    with open(train_json_path, 'r') as f:
        train_predictions = json.load(f)
    
    with open(test_json_path, 'r') as f:
        test_predictions = json.load(f)
    
    print("Preparing train image mappings...")
    # 학습 이미지 경로 및 레이블 매핑
    train_dataset = EnhancedImageDataset(
        data_dir=TRAIN_IMG_DIR,
        processor=None,
        mode='train',
        model_type='clip'
    )
    
    train_images = {}
    train_labels = {}
    
    for i in range(len(train_dataset)):
        try:
            info = train_dataset.get_img_info(i)
            img_id = info['image_id']
            img_path = info['abs_path']
            label = info['label_text']
            
            train_images[img_id] = img_path
            train_labels[img_id] = label
        except:
            continue
    
    print("Preparing test image mappings...")
    # 테스트 이미지 경로 매핑
    test_csv_path = os.path.join(DATA_ROOT_DIR, 'test.csv')
    test_df = pd.read_csv(test_csv_path)
    
    test_images = {}
    
    for _, row in test_df.iterrows():
        img_id = row['ID']
        img_path = row['img_path']
        
        if img_path.startswith('./'):
            img_path = img_path[2:]
        
        full_img_path = os.path.join(DATA_ROOT_DIR, img_path)
        test_images[img_id] = full_img_path
    
    print(f"✅ Loaded {len(train_images)} train images")
    print(f"✅ Loaded {len(test_images)} test images")
    print(f"✅ Loaded {len(train_predictions)} train predictions")
    print(f"✅ Loaded {len(test_predictions)} test predictions")
    
    # 실제 파이프라인 실행
    quick_clip_pipeline(train_images, train_labels, train_predictions,
                       test_images, test_predictions, args)