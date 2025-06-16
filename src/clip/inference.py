import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from transformers import CLIPProcessor

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.config.config import CFG, DEVICE, DATA_ROOT_DIR, TEST_DATA_DIR, TRAIN_IMG_DIR
from src.utils.utils import collate_fn_skip_none
from src.clip.dataset import CustomImageDataset
from src.clip.model import CLIPBasedModel

def generate_train_top_k(args, model, processor, class_names, num_classes, train_data_path, output_json_path):
    """
    학습 데이터에 대한 Top-K 예측 JSON 생성 (개별 이미지 처리 방식)
    """
    print("Generating Top-K predictions for train data...")
    
    try:
        dataset = CustomImageDataset(
            data_dir=train_data_path,
            processor=processor,
            mode='train'
        )
    except Exception as e:
        print(f"Error creating train dataset: {e}")
        return

    if len(dataset) == 0:
        print("Error: Train dataset is empty.")
        return

    model.eval()
    all_top_k_predictions = {}
    
    # 개별 이미지 처리로 순서 불일치 문제 해결
    for idx in tqdm(range(len(dataset)), desc="Processing train images"):
        try:
            item = dataset[idx]
            if item is None:
                continue
            
            # 고유 ID 생성 (클래스명 포함하여 중복 방지)
            img_path, label = dataset.samples[idx]
            class_name = os.path.basename(os.path.dirname(img_path))
            file_name = os.path.splitext(os.path.basename(img_path))[0]  # 확장자 제거
            unique_id = file_name
            
            # 개별 예측
            pixel_values = item['pixel_values'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(pixel_values)
                probs = torch.softmax(outputs, dim=1)
                probs_array = probs.cpu().numpy()[0]
            
            # Top-K 추출
            top_k_indices = np.argsort(probs_array)[-args.top_k:][::-1]
            top_k_info_for_img = []
            
            for k_idx in top_k_indices:
                class_name_str = class_names[k_idx]
                confidence = probs_array[k_idx]
                top_k_info_for_img.append({
                    "class_name": class_name_str, 
                    "confidence": round(float(confidence), 6)
                })
            
            all_top_k_predictions[unique_id] = top_k_info_for_img
            
        except Exception as e:
            print(f"Error processing train image {idx}: {e}")
            continue

    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_top_k_predictions, f, ensure_ascii=False, indent=4)
        print(f"Top-K predictions for train data saved to {output_json_path}")
        print(f"Total train predictions: {len(all_top_k_predictions)}")
    except Exception as e:
        print(f"Error saving Top-K JSON for train data: {e}")

def generate_test_inference(args, model, processor, class_names, num_classes, test_csv_path, output_json_path):
    """
    테스트 데이터에 대한 추론 (submission CSV + Top-K JSON 생성)
    """
    print("Generating predictions for test data...")
    
    # 테스트 데이터 ID 순서 로드
    try:
        test_df_for_id_order = pd.read_csv(test_csv_path)
        ordered_test_ids = test_df_for_id_order['ID'].tolist()
    except Exception as e:
        print(f"Error reading test.csv: {e}")
        return

    try:
        test_dataset = CustomImageDataset(
            data_dir=args.data_root_dir,
            processor=processor,
            mode='test',
            test_csv_path=test_csv_path
        )
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        return

    if len(test_dataset) == 0:
        print("Error: Test dataset is empty.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_skip_none
    )

    model.eval()
    all_test_probs_list = []
    all_test_ids_list = []

    # 배치별 처리
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Test inference")):
            if not batch_data:
                continue
            
            pixel_values = batch_data.get('pixel_values')
            img_ids_batch = batch_data.get('img_id')

            if pixel_values is None or img_ids_batch is None:
                print(f"Warning: Batch {batch_idx} has missing data, skipping.")
                continue
                
            pixel_values = pixel_values.to(DEVICE)
            outputs = model(pixel_values)
            probs = torch.softmax(outputs, dim=1)

            all_test_probs_list.append(probs.cpu().numpy())
            all_test_ids_list.extend(img_ids_batch)

    if not all_test_probs_list:
        print("Error: No predictions were made for test data.")
        return
        
    all_test_probs_np = np.concatenate(all_test_probs_list, axis=0)

    # 1. Submission CSV 생성
    submission_df = pd.DataFrame(all_test_probs_np, columns=class_names)
    submission_df.insert(0, 'ID', all_test_ids_list)

    # ID 타입 통일 (중요!)
    submission_df['ID'] = submission_df['ID'].astype(str)
    ordered_test_ids_str = [str(id_val) for id_val in ordered_test_ids]

    try:
        submission_df = submission_df.set_index('ID').reindex(ordered_test_ids_str).reset_index()
        
        # NaN 확인
        if submission_df[class_names].isnull().any().any():
            missing_ids = submission_df[submission_df[class_names].isnull().any(axis=1)]['ID'].tolist()
            print(f"Warning: Missing predictions for {len(missing_ids)} IDs: {missing_ids[:5]}...")
            
    except KeyError as e:
        print(f"Error reindexing submission DataFrame: {e}")
        print("Available IDs:", submission_df['ID'].head().tolist())
        print("Expected IDs:", ordered_test_ids_str[:5])
        print("Saving without reordering by test.csv ID order.")

    # Submission CSV 저장
    try:
        submission_csv_path = args.submission_save_path
        os.makedirs(os.path.dirname(submission_csv_path), exist_ok=True)
        submission_df.to_csv(submission_csv_path, index=False)
        print(f"Submission file saved to {submission_csv_path}")
        print(f"Submission shape: {submission_df.shape}")
    except Exception as e:
        print(f"Error saving submission file: {e}")

    # 2. Top-K JSON 생성
    all_top_k_predictions = {}
    
    # submission_df를 사용하여 순서가 맞는 Top-K 생성
    temp_submission_for_topk = submission_df.set_index('ID')
    
    for img_id in tqdm(ordered_test_ids_str, desc="Extracting Top-K for test"):
        if img_id not in temp_submission_for_topk.index:
            print(f"Warning: ID {img_id} not found in predictions for Top-K. Skipping.")
            continue
            
        try:
            probs_for_img_series = temp_submission_for_topk.loc[img_id, class_names]
            probs_for_img = probs_for_img_series.values.astype(float)
            
            top_k_indices = np.argsort(probs_for_img)[-args.top_k:][::-1]
            top_k_info_for_img = []
            
            for k_idx in top_k_indices:
                class_name_str = class_names[k_idx]
                confidence = probs_for_img[k_idx]
                top_k_info_for_img.append({
                    "class_name": class_name_str, 
                    "confidence": round(float(confidence), 6)
                })
            
            all_top_k_predictions[img_id] = top_k_info_for_img
            
        except Exception as e:
            print(f"Error processing Top-K for ID {img_id}: {e}")
            continue

    # Top-K JSON 저장
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_top_k_predictions, f, ensure_ascii=False, indent=4)
        print(f"Top-K predictions for test data saved to {output_json_path}")
        print(f"Total test predictions: {len(all_top_k_predictions)}")
    except Exception as e:
        print(f"Error saving Top-K JSON for test data: {e}")

def main(args):
    print(f"Using device: {DEVICE}")
    print(f"Running with mode: {args.mode}")

    # 프로세서 로드
    try:
        processor = CLIPProcessor.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Failed to load CLIPProcessor for {args.model_name}: {e}")
        return

    # 클래스 이름 로드
    class_names = None
    num_classes = CFG['NUM_CLASSES']
    sample_submission_path = os.path.join(args.data_root_dir, 'sample_submission.csv')
    
    try:
        if not os.path.exists(sample_submission_path):
            raise FileNotFoundError(f"sample_submission.csv not found at {sample_submission_path}")
        
        sample_df = pd.read_csv(sample_submission_path)
        class_names = sample_df.columns[1:].tolist()
        num_classes = len(class_names)
        print(f"Loaded {num_classes} class names from sample_submission.csv.")
        
    except Exception as e:
        print(f"Warning: Could not load class names from sample_submission.csv: {e}")
        print("Trying to load from train_img_dir_for_meta...")
        
        try:
            if not os.path.isdir(args.train_img_dir_for_meta):
                raise FileNotFoundError(f"Directory not found: {args.train_img_dir_for_meta}")
            
            temp_ds_for_meta = CustomImageDataset(
                data_dir=args.train_img_dir_for_meta, 
                processor=processor, 
                mode='train'
            )
            class_names = temp_ds_for_meta.classes
            num_classes = len(class_names)
            print(f"Loaded {num_classes} class names from {args.train_img_dir_for_meta}.")
            
        except Exception as e1:
            print(f"Error: Could not load class names from any source: {e1}")
            return

    if not class_names:
        print("Fatal Error: No class names could be loaded.")
        return

    print(f"Using {num_classes} classes: {class_names[:3]}... (Total: {len(class_names)})")

    # 모델 로드
    try:
        model = CLIPBasedModel(model_name=args.model_name, num_classes=num_classes)
        model.load_state_dict(torch.load(args.model_load_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded from {args.model_load_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 실행 모드에 따라 분기
    if args.mode == 'test_inference':
        test_csv_file = os.path.join(args.data_root_dir, 'test.csv')
        if not os.path.exists(test_csv_file):
            print(f"Error: test.csv not found at {test_csv_file}")
            return
        
        # 출력 파일 경로 설정
        if args.submission_save_path:
            base_name = os.path.splitext(os.path.basename(args.submission_save_path))[0]
            dir_name = os.path.dirname(args.submission_save_path)
            test_top_k_json_path = os.path.join(dir_name, f"{base_name}_top_{args.top_k}.json")
        else:
            test_top_k_json_path = os.path.join(args.output_dir, f"test_clip_model_A_top_{args.top_k}.json")
            args.submission_save_path = os.path.join(args.output_dir, "submission_clip_model_A.csv")

        generate_test_inference(
            args, model, processor, class_names, num_classes,
            test_csv_file, test_top_k_json_path
        )

    elif args.mode == 'train_top_k_generation':
        train_data_path = args.train_img_dir
        if not os.path.isdir(train_data_path):
            print(f"Error: Training image directory not found at {train_data_path}")
            return

        train_top_k_json_path = os.path.join(args.output_dir, f"train_clip_model_A_top_{args.top_k}.json")
        
        generate_train_top_k(
            args, model, processor, class_names, num_classes,
            train_data_path, train_top_k_json_path
        )
    else:
        print(f"Error: Unsupported mode '{args.mode}'. Choose 'test_inference' or 'train_top_k_generation'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference and Top-K generation for CLIP-based model.")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['test_inference', 'train_top_k_generation'],
                        help="Execution mode")
    
    # 공통 인자
    parser.add_argument('--model_name', type=str, default=CFG.get('MODEL_NAME', 'openai/clip-vit-base-patch32'))
    parser.add_argument('--data_root_dir', type=str, default=DATA_ROOT_DIR)
    parser.add_argument('--train_img_dir_for_meta', type=str, default=TRAIN_IMG_DIR)
    parser.add_argument('--batch_size', type=int, default=CFG.get('BATCH_SIZE', 64))
    parser.add_argument('--num_workers', type=int, default=os.cpu_count()//2 if os.cpu_count() and os.cpu_count() > 1 else 0)
    parser.add_argument('--model_load_path', type=str, default=CFG.get('MODEL_SAVE_PATH', '../models/best_model.pth'))
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='../../outputs')

    parser.add_argument('--submission_save_path', type=str, 
                        default=CFG.get('SUBMISSION_PATH', '../../outputs/submission_model_A.csv'))
    
    # train_top_k_generation 모드 전용
    parser.add_argument('--train_img_dir', type=str, default=TRAIN_IMG_DIR)

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)