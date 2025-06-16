import torch
import json
import numpy as np
import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import pandas as pd
import argparse
from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset, DataLoader
import sys
import gc

# 환경 변수 설정 (경고 제거)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_ROOT_DIR, TRAIN_IMG_DIR
from utils.data.enhanced_dataset import EnhancedImageDataset

class BatchImageDataset(Dataset):
    """배치 처리를 위한 이미지 데이터셋 - 메모리 최적화 버전"""
    def __init__(self, image_paths: List[str], image_ids: List[str], predictions_dict: Dict[str, List[Dict]], 
                 base_prompt: str = "Analyze this car image and describe the vehicle's features in detail.",
                 max_image_size: tuple = (768, 768)):  # 이미지 크기 제한 추가
        self.image_paths = image_paths
        self.image_ids = image_ids
        self.predictions_dict = predictions_dict
        self.base_prompt = base_prompt
        self.max_image_size = max_image_size
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        """메모리 최적화 + 품질 유지된 데이터 로딩"""
        img_path = self.image_paths[idx]
        img_id = self.image_ids[idx]
        
        try:
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found at {img_path}. Skipping.")
                return None
            
            # 이미지 크기 제한으로 메모리 사용량 감소
            image = Image.open(img_path).convert("RGB")
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # 개선된 프롬프트 생성 로직
            predictions = self.predictions_dict.get(img_id, [])
            if predictions:
                top_predictions = predictions[:3]
                pred_info_parts = []
                for pred in top_predictions:
                    class_name = pred.get('class_name', 'unknown')
                    confidence = pred.get('confidence', 0.0)
                    pred_info_parts.append(f"{class_name}({confidence:.0%})")
                
                pred_info_str = ", ".join(pred_info_parts)
                
                # 균형잡힌 프롬프트 (메모리 절약 + 성능 유지)
                prompt = (f"Analyze this car's visual features and identify the model. "
                        f"CV predictions: {pred_info_str}. "
                        f"Describe brand markers, body style, and design elements.")
                
            else:
                prompt = "Analyze this car image. Describe the brand, model, and distinctive visual characteristics."
                
            return {
                'image': image,
                'prompt': prompt,
                'image_id': img_id,
                'image_path': img_path
            }
        except Exception as e:
            print(f"Error processing image {img_path}: {e}. Skipping.")
            return None

def collate_batch_fn_optimized(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """메모리 최적화된 배치 콜레이트 함수"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
        
    return {
        'images': [item['image'] for item in batch],
        'prompts': [item['prompt'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'image_paths': [item['image_path'] for item in batch]
    }

class QwenVLBatchFeatureExtractor:
    """메모리 최적화된 Qwen2.5-VL 특징 추출기"""
    def __init__(self, 
                 model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 use_awq: bool = False,
                 batch_size: int = 2,  # 기본값을 2로 감소
                 device: str = "auto",
                 enable_memory_optimization: bool = True):
        
        print(f"🚀 Loading Qwen2.5-VL model: {model_id} (AWQ: {use_awq}, Batch size: {batch_size})")
        
        self.batch_size = batch_size
        self.enable_memory_optimization = enable_memory_optimization
        
        # 메모리 최적화 설정
        if enable_memory_optimization:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        model_load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": device
        }

        # AWQ 모델 로드 (경고 무시)
        if use_awq:
            awq_models = {
                "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", 
                "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
                "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
            }
            
            target_model_id = awq_models.get(model_id, model_id)
            if model_id != target_model_id or model_id.endswith("-AWQ"):
                print(f"🚀 Loading AWQ quantized model: {target_model_id}")
                try:
                    import warnings
                    warnings.filterwarnings("ignore", category=DeprecationWarning)  # AWQ 경고 무시
                    
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        target_model_id, **model_load_kwargs
                    )
                    print("✅ AWQ model loaded successfully.")
                except Exception as e:
                    print(f"❌ Failed to load AWQ model: {e}")
                    print("⚠️  Falling back to standard model.")
                    use_awq = False
            else:
                print("⚠️  No AWQ model found, using standard model.")
                use_awq = False
        
        # 표준 모델 로드
        if not use_awq:
            print(f"🚀 Loading standard model: {model_id}")
            try:
                model_load_kwargs["attn_implementation"] = "flash_attention_2"
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_load_kwargs
                )
                print("✅ Standard model with FlashAttention 2 loaded.")
            except:
                print("⚠️  Loading without FlashAttention 2.")
                del model_load_kwargs["attn_implementation"]
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_load_kwargs
                )
        
        self.device = self.model.device if hasattr(self.model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        # 메모리 최적화: gradient 비활성화
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"✅ Qwen2.5-VL model ready on {self.device}")
        
        # 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()
    
    def extract_batch_features(self, batch_data: Dict) -> List[tuple[str, np.ndarray]]:
        """메모리 효율적인 배치 특징 추출"""
        if batch_data is None or not batch_data['images']:
            return []
            
        images = batch_data['images']
        prompts = batch_data['prompts']
        image_ids = batch_data['image_ids']
        
        try:
            # 메모리 정리
            if self.enable_memory_optimization:
                gc.collect()
                torch.cuda.empty_cache()
            
            batch_messages = []
            for image, prompt in zip(images, prompts):
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }]
                batch_messages.append(messages)
            
            batch_texts = []
            all_image_inputs = []
            
            for messages in batch_messages:
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text_input)
                
                image_inputs_for_single_item, _ = process_vision_info(messages)
                all_image_inputs.extend(image_inputs_for_single_item)
            
            # 프로세서로 배치 처리 (메모리 최적화)
            inputs = self.processor(
                text=batch_texts,
                images=all_image_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,  # 메모리 절약을 위해 True로 변경
                max_length=1024,  # 최대 길이 제한
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 배치 추론
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # 특징 추출 (마지막 hidden state)
                features = outputs.hidden_states[-1]
                
                # 어텐션 마스크를 사용한 평균 풀링
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                pooled_features = (features * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
                
                # CPU로 이동하여 numpy 변환
                features_np = pooled_features.cpu().numpy()
                
                # 메모리 정리
                del outputs, features, pooled_features, inputs
                if self.enable_memory_optimization:
                    torch.cuda.empty_cache()
                
            return [(fid, feat) for fid, feat in zip(image_ids, features_np)]
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM in batch {image_ids}: {e}")
            print("🔄 Clearing cache and returning dummy features...")
            
            # 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            # 더미 특징 반환
            dummy_size = getattr(self.model.config, 'hidden_size', 3584)
            return [(fid, np.zeros(dummy_size, dtype=np.float32)) for fid in image_ids]
            
        except Exception as e:
            print(f"Error processing batch {image_ids}: {e}")
            dummy_size = getattr(self.model.config, 'hidden_size', 3584)
            return [(fid, np.zeros(dummy_size, dtype=np.float32)) for fid in image_ids]

def qwen_vl_optimized_pipeline(train_images: Dict[str, str], train_labels: Dict[str, str], train_predictions: Dict[str, List[Dict]], 
                              test_images: Dict[str, str], test_predictions: Dict[str, List[Dict]], args: argparse.Namespace):
    """메모리 최적화된 Qwen2.5-VL 파이프라인"""
    
    # 메모리 최적화된 특징 추출기 초기화
    extractor = QwenVLBatchFeatureExtractor(
        model_id=args.model_id,
        use_awq=args.use_awq,
        batch_size=args.batch_size,
        device="auto",
        enable_memory_optimization=True
    )
    
    # Train 처리만 필요한 경우 확인
    if args.train_only and os.path.exists(args.train_output):
        print(f"✅ Train features already exist at {args.train_output}. Skipping train processing.")
    else:
        # 학습 데이터 처리
        print("🔄 Preparing training data...")
        train_img_paths = []
        train_img_ids = []
        train_img_labels = []
        
        matched_count = 0
        for pred_id in list(train_predictions.keys()):
            img_id_with_ext = pred_id + ".jpg"
            
            if img_id_with_ext in train_images and img_id_with_ext in train_labels:
                img_path = train_images[img_id_with_ext]
                if os.path.exists(img_path):
                    train_img_paths.append(img_path)
                    train_img_ids.append(pred_id)
                    train_img_labels.append(train_labels[img_id_with_ext])
                    matched_count += 1
        
        if train_img_paths:
            print(f"✅ Prepared {matched_count} training samples.")
            
            # 메모리 최적화된 데이터셋
            train_dataset = BatchImageDataset(
                train_img_paths, train_img_ids, train_predictions,
                max_image_size=(512, 512)  # 이미지 크기 제한
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=min(args.num_workers, 2),  # worker 수 제한
                collate_fn=collate_batch_fn_optimized,
                pin_memory=False  # 메모리 절약
            )
            
            print("🚀 Extracting training features...")
            train_features = []
            train_ids_final = []
            labels_final = []
            
            for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Processing train batches")):
                if batch_data is None:
                    continue
                    
                # 주기적 메모리 정리
                if batch_idx % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                batch_results = extractor.extract_batch_features(batch_data)
                for img_id, features in batch_results:
                    if features is not None and features.size > 0:
                        train_features.append(features)
                        train_ids_final.append(img_id)
                        try:
                            original_idx = train_img_ids.index(img_id)
                            labels_final.append(train_img_labels[original_idx])
                        except ValueError:
                            labels_final.append("unknown")
            
            if train_features:
                print(f"💾 Saving {len(train_features)} train features...")
                np.savez(args.train_output,
                         features=np.vstack(train_features),
                         image_ids=train_ids_final,
                         labels=labels_final)
                print(f"📁 Train features saved: {args.train_output}")
                print(f"   Shape: {np.vstack(train_features).shape}")
    
    if not args.train_only:
        if os.path.exists(args.test_output):
            print(f"✅ Test features already exist at {args.test_output}. Skipping test processing.")
        else:
            print("🔄 Preparing test data...")
            test_img_paths = []
            test_img_ids = []
            
            for pred_id in list(test_predictions.keys()):
                if pred_id in test_images:
                    img_path = test_images[pred_id]
                    if os.path.exists(img_path):
                        test_img_paths.append(img_path)
                        test_img_ids.append(pred_id)
                        
            if test_img_paths:
                print(f"✅ Prepared {len(test_img_paths)} test samples.")
                
                test_dataset = BatchImageDataset(
                    test_img_paths, test_img_ids, test_predictions,
                    max_image_size=(512, 512)
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=min(args.num_workers, 2),
                    collate_fn=collate_batch_fn_optimized,
                    pin_memory=False
                )
                
                print("🚀 Extracting test features...")
                test_features = []
                test_ids_final = []
                
                for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Processing test batches")):
                    if batch_data is None:
                        continue
                        
                    if batch_idx % 100 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    batch_results = extractor.extract_batch_features(batch_data)
                    for img_id, features in batch_results:
                        if features is not None and features.size > 0:
                            test_features.append(features)
                            test_ids_final.append(img_id)
                
                if test_features:
                    print(f"💾 Saving {len(test_features)} test features...")
                    np.savez(args.test_output,
                             features=np.vstack(test_features),
                             image_ids=test_ids_final)
                    print(f"📁 Test features saved: {args.test_output}")
                    print(f"   Shape: {np.vstack(test_features).shape}")
    
    print("🎉 Processing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-optimized Qwen2.5-VL feature extraction")
    
    parser.add_argument('--train_json', type=str, 
                       default='outputs/train_teammate_convnext_top_5.json',
                       help='Path to train top-k JSON file')
    parser.add_argument('--test_json', type=str,
                       default='outputs/test_teammate_convnext_top_5.json',
                       help='Path to test top-k JSON file')
    parser.add_argument('--train_output', type=str,
                       default='outputs/qwen_vlm_batch_train_features.npz',
                       help='Path to save train features')
    parser.add_argument('--test_output', type=str,
                       default='outputs/qwen_vlm_batch_test_features.npz',
                       help='Path to save test features')
    parser.add_argument('--model_id', type=str,
                       default='Qwen/Qwen2.5-VL-3B-Instruct',
                       help='Hugging Face model ID for Qwen2.5-VL')
    parser.add_argument('--use_awq', action='store_true',
                       help='Use AWQ quantization for faster inference')
    parser.add_argument('--batch_size', type=int, default=2,  # 기본값 감소
                       help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=1,  # 기본값 감소
                       help='Number of workers for data loading')
    parser.add_argument('--train_only', action='store_true',
                       help='Process train data only')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting optimized feature extraction...")
    print(f"   Model: {args.model_id}")
    print(f"   AWQ: {args.use_awq}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Workers: {args.num_workers}")
    print(f"   Train only: {args.train_only}")
    
    # JSON 파일들 로드
    try:
        with open(args.train_json, 'r') as f:
            train_predictions = json.load(f)
        print(f"✅ Loaded {len(train_predictions)} train predictions")
    except Exception as e:
        print(f"Error loading train JSON: {e}")
        sys.exit(1)
    
    try:
        with open(args.test_json, 'r') as f:
            test_predictions = json.load(f)
        print(f"✅ Loaded {len(test_predictions)} test predictions")
    except Exception as e:
        print(f"Error loading test JSON: {e}")
        sys.exit(1)
    
    # 이미지 매핑 준비
    print("🔄 Preparing image mappings...")
    
    train_dataset_info = EnhancedImageDataset(
        data_dir=TRAIN_IMG_DIR,
        processor=None, 
        mode='train',
        model_type='qwen_vl'
    )
    
    train_images = {}
    train_labels = {}
    
    for i in tqdm(range(len(train_dataset_info)), desc="Mapping train images"):
        try:
            info = train_dataset_info.get_img_info(i)
            img_id_with_ext = info['image_id']
            img_path = info['abs_path']
            label = info['label_text']
            
            train_images[img_id_with_ext] = img_path
            train_labels[img_id_with_ext] = label
        except Exception as e:
            continue
    
    # 테스트 이미지 매핑
    test_csv_path = os.path.join(DATA_ROOT_DIR, 'test.csv')
    test_df = pd.read_csv(test_csv_path)
        
    test_images = {}
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Mapping test images"):
        img_id_no_ext = row['ID']
        img_path_rel = row['img_path']
        if img_path_rel.startswith('./'):
            img_path_rel = img_path_rel[2:]
        full_img_path = os.path.join(DATA_ROOT_DIR, img_path_rel)
        test_images[img_id_no_ext] = full_img_path
    
    print(f"✅ Mapped {len(train_images)} train images")
    print(f"✅ Mapped {len(test_images)} test images")
    
    # 최적화된 파이프라인 실행
    qwen_vl_optimized_pipeline(train_images, train_labels, train_predictions,
                              test_images, test_predictions, args)