import os
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Project imports
from src.config.config import CFG, DEVICE, DATA_ROOT_DIR, TEST_DATA_DIR, TRAIN_IMG_DIR
from src.utils.utils import collate_fn_skip_none
from src.utils.data.enhanced_dataset import EnhancedImageDataset
from src.utils.data.enhanced_model import EnhancedModel, get_model_and_processor

def load_model_smart_patch(model, model_path, model_type, model_name, num_classes, device):
    """
    스마트 모델 로더 - 기존 enhanced_inference.py용 패치
    """
    print(f"🔄 Smart loading model from: {model_path}")
    
    try:
        # 1. 모델 파일 로드 및 구조 분석
        state_dict = torch.load(model_path, map_location='cpu')
        sample_keys = list(state_dict.keys())[:10]
        
        print(f"📊 Found {len(state_dict)} parameters")
        print(f"🔍 Sample keys: {sample_keys[:3]}...")
        
        # 2. 구조 타입 감지
        is_timm_based = any('vision_backbone' in key for key in sample_keys)
        is_pure_timm = any('backbone.features' in key or 'features.' in key for key in sample_keys)
        
        print(f"Model structure - TimmBased: {is_timm_based}, Pure timm: {is_pure_timm}")
        
        # 3. 적절한 방식으로 로딩
        if is_timm_based:
            # 기존 TimmBasedModel 방식
            model.load_state_dict(state_dict)
            print(f"✅ Loaded as TimmBasedModel")
            
        elif is_pure_timm:
            # 순수 timm 모델을 TimmBasedModel 형태로 변환 로딩
            import timm
            pure_timm_model = timm.create_model(
                model_name, 
                pretrained=False, 
                num_classes=num_classes
            )
            pure_timm_model.load_state_dict(state_dict)
            
            # TimmBasedModel의 vision_backbone에 순수 timm 모델 할당
            model.vision_backbone = pure_timm_model
            print(f"✅ Loaded pure timm model into TimmBasedModel wrapper")
            
        else:
            raise Exception(f"Unknown model structure. Sample keys: {sample_keys[:5]}")
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Smart loading failed: {e}")
        raise

def generate_train_top_k(args, model, dataset, class_names, output_json_path):
    """
    Generate Top-K predictions for training data
    """
    print(f"Generating Top-K predictions for train data...")
    
    model.eval()
    all_top_k_predictions = {}
    
    for idx in tqdm(range(len(dataset)), desc="Processing train images"):
        try:
            item = dataset[idx]
            if item is None:
                continue
            
            # Get image info
            img_info = dataset.get_img_info(idx)
            unique_id = os.path.splitext(os.path.basename(img_info['abs_path']))[0]  # Remove extension
            
            # Get prediction
            pixel_values = item['pixel_values'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(pixel_values)
                probs = torch.softmax(outputs, dim=1)
                probs_array = probs.cpu().numpy()[0]
            
            # Get Top-K predictions
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

    # Save to JSON
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_top_k_predictions, f, ensure_ascii=False, indent=4)
        print(f"Top-K predictions for train data saved to {output_json_path}")
        print(f"Total train predictions: {len(all_top_k_predictions)}")
    except Exception as e:
        print(f"Error saving Top-K JSON for train data: {e}")

def generate_test_inference(args, model, test_dataset, class_names, output_json_path):
    """
    Generate inference for test data
    """
    print("Generating predictions for test data...")
    
    # Load test.csv for ID order
    test_csv_path = os.path.join(args.data_root_dir, 'test.csv')
    test_df = pd.read_csv(test_csv_path)
    ordered_test_ids = test_df['ID'].tolist()
    
    # Create test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_skip_none
    )
    
    # Make predictions
    model.eval()
    all_test_probs_list = []
    all_test_ids_list = []
    
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
    
    # Create submission CSV
    submission_df = pd.DataFrame(all_test_probs_np, columns=class_names)
    submission_df.insert(0, 'ID', all_test_ids_list)
    
    # Ensure proper ID type and ordering
    submission_df['ID'] = submission_df['ID'].astype(str)
    ordered_test_ids_str = [str(id_val) for id_val in ordered_test_ids]
    
    # Sort by test.csv order
    try:
        submission_df = submission_df.set_index('ID').reindex(ordered_test_ids_str).reset_index()
        
        # Check for NaN values
        if submission_df[class_names].isnull().any().any():
            missing_ids = submission_df[submission_df[class_names].isnull().any(axis=1)]['ID'].tolist()
            print(f"Warning: Missing predictions for {len(missing_ids)} IDs")
    except KeyError as e:
        print(f"Error reindexing submission DataFrame: {e}")
        print("Saving without reordering by test.csv ID order.")
    
    # Save submission CSV
    submission_csv_path = args.submission_save_path
    os.makedirs(os.path.dirname(submission_csv_path), exist_ok=True)
    submission_df.to_csv(submission_csv_path, index=False)
    print(f"Submission file saved to {submission_csv_path}")
    
    # Create Top-K JSON
    all_top_k_predictions = {}
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
    
    # Save Top-K JSON
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
    
    # Get class names from sample submission
    sample_submission_path = os.path.join(args.data_root_dir, 'sample_submission.csv')
    if os.path.exists(sample_submission_path):
        sample_df = pd.read_csv(sample_submission_path)
        class_names = sample_df.columns[1:].tolist()
        num_classes = len(class_names)
        print(f"Loaded {num_classes} class names from sample_submission.csv")
    else:
        print(f"Warning: sample_submission.csv not found at {sample_submission_path}")
        print("Using config NUM_CLASSES instead")
        num_classes = args.num_classes
        class_names = [f"class_{i}" for i in range(num_classes)]  # Dummy class names
    
    # Get model and processor
    model, processor = get_model_and_processor(
        model_type=args.model_type,
        model_name=args.model_name,
        num_classes=num_classes,
        freeze_backbone=False  # No need to freeze for inference
    )
    
    # Load model weights (Smart Loading)
    try:
        model = load_model_smart_patch(
            model=model,
            model_path=args.model_load_path,
            model_type=args.model_type,
            model_name=args.model_name,
            num_classes=num_classes,
            device=DEVICE
        )
        print(f"Model loaded from {args.model_load_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Execute based on mode
    if args.mode == 'train_top_k_generation':
        # Create dataset for training data
        train_dataset = EnhancedImageDataset(
            data_dir=args.train_img_dir,
            processor=processor,
            mode='train',
            model_type=args.model_type
        )
        
        # Generate output path
        output_json_path = os.path.join(args.output_dir, f"train_{args.model_type}_{args.model_name.split('/')[-1]}_top_{args.top_k}.json")
        
        # Generate Top-K predictions
        generate_train_top_k(
            args, model, train_dataset, class_names, output_json_path
        )
    
    elif args.mode == 'test_inference':
        # Create dataset for test data
        test_dataset = EnhancedImageDataset(
            data_dir=args.data_root_dir,
            processor=processor,
            mode='test',
            test_csv_path=os.path.join(args.data_root_dir, 'test.csv'),
            model_type=args.model_type
        )
        
        # Generate output paths
        model_name_short = args.model_name.split('/')[-1]
        test_top_k_json_path = os.path.join(args.output_dir, f"test_{args.model_type}_{model_name_short}_top_{args.top_k}.json")
        
        # Generate test inference and Top-K predictions
        generate_test_inference(
            args, model, test_dataset, class_names, test_top_k_json_path
        )
    
    else:
        print(f"Error: Unsupported mode '{args.mode}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced inference for CV models")
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train_top_k_generation', 'test_inference'],
                        help="Execution mode")
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='clip',
                        choices=['clip', 'vit', 'swin', 'timm'],
                        help="Type of model to use")
    
    parser.add_argument('--model_name', type=str, default=CFG['MODEL_NAME'],
                        help="Model name (from HuggingFace or timm)")
    
    parser.add_argument('--num_classes', type=int, default=CFG['NUM_CLASSES'],
                        help="Number of classes to predict")
    
    parser.add_argument('--model_load_path', type=str, 
                        default=CFG['MODEL_SAVE_PATH'],
                        help="Path to the saved model weights")
    
    # Data parameters
    parser.add_argument('--data_root_dir', type=str, default=DATA_ROOT_DIR,
                        help="Root directory for data")
    
    parser.add_argument('--train_img_dir', type=str, default=TRAIN_IMG_DIR,
                        help="Directory containing training images")
    
    parser.add_argument('--batch_size', type=int, default=CFG['BATCH_SIZE'],
                        help="Batch size for inference")
    
    parser.add_argument('--top_k', type=int, default=5,
                        help="Number of top predictions to include")
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help="Directory to save outputs")
    
    parser.add_argument('--submission_save_path', type=str, 
                        default=CFG['SUBMISSION_PATH'],
                        help="Path to save the submission file")
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=os.cpu_count()//2,
                        help="Number of workers for DataLoader")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)