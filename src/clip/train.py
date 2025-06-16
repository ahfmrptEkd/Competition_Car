import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import log_loss

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.config.config import CFG, DEVICE, TRAIN_IMG_DIR
from src.utils.utils import seed_everything, collate_fn_skip_none, calculate_log_loss
from src.clip.dataset import CustomImageDataset
from src.clip.model import CLIPBasedModel

def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    train_loss_sum = 0.0
    correct = 0
    total = 0

    train_pbar = tqdm(train_loader, desc=f"Training Epoch")
    for batch_data in train_pbar:
        if not batch_data: continue

        pixel_values = batch_data['pixel_values'].to(device)
        labels = batch_data['label'].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        train_loss_sum += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        current_lr = optimizer.param_groups[0]['lr']
        accuracy = 100. * correct / total
        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy:.2f}%",
            'lr': f"{current_lr:.2e}"
        })
        
    return train_loss_sum / len(train_loader), 100. * correct / total

def validate_one_epoch(model, val_loader, criterion, device, num_classes_for_logloss):
    model.eval()
    val_loss_sum = 0.0
    all_val_probs = []
    all_val_labels = []
    correct = 0
    total = 0

    val_pbar = tqdm(val_loader, desc=f"Validation Epoch")
    with torch.no_grad():
        for batch_data in val_pbar:
            if not batch_data: continue

            pixel_values = batch_data['pixel_values'].to(device)
            labels = batch_data['label'].to(device)
            
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            probs = torch.softmax(outputs, dim=1)
            all_val_probs.extend(probs.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
            
            accuracy = 100. * correct / total
            val_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.2f}%"
            })
    
    avg_val_loss = val_loss_sum / len(val_loader)
    val_logloss = log_loss(all_val_labels, np.array(all_val_probs), list(range(num_classes_for_logloss)))
    final_accuracy = 100. * correct / total

    return avg_val_loss, val_logloss, final_accuracy

def main(args):
    seed_everything(CFG['SEED'])
    print(f"Using device: {DEVICE}")
    print(f"Running with config: {args}")

    # processor 코드
    try:
        processor = CLIPProcessor.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Failed to load CLIPProcessor for {args.model_name}: {e}")
        return
    
    try:
        temp_ds_for_meta = CustomImageDataset(data_dir=TRAIN_IMG_DIR, processor=processor, mode='train')
        class_names = temp_ds_for_meta.classes
        class_to_idx = temp_ds_for_meta.class_to_idx
        num_classes = len(class_names)
        print(f"Number of classes: {num_classes}")
        if num_classes == 0:
             print(f"Error: No classes found in {args.train_img_dir}. Check the directory structure.")
             return
    except Exception as e:
        print(f"Error initializing dataset for metadata: {e}")
        return

    # transform은 None으로 하고, Subset에서 각각의 transform을 적용하는 CustomImageDataset 인스턴스를 새로 만듭니다.
    all_train_samples_dataset = CustomImageDataset(
        data_dir=args.train_img_dir, processor=processor, mode='train',
        class_names=class_names, class_to_idx=class_to_idx
    )

    # None인 샘플 필터링 (파일 경로 오류 등)
    valid_indices = [i for i, sample in enumerate(all_train_samples_dataset) if sample is not None]
    if not valid_indices:
        print(f"Error: No valid samples found in {args.train_img_dir}.")
        return
        
    targets = [all_train_samples_dataset[i]['label'].item() for i in valid_indices]


    # Stratified Split
    train_indices, val_indices = train_test_split(
        valid_indices, # 전체 인덱스 대신 유효한 인덱스 사용
        test_size=args.val_split,
        stratify=targets, # 유효한 타겟으로 stratify
        random_state=args.seed
    )
    print(f'Total valid samples: {len(valid_indices)}')


    train_augment = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
    ])
    val_augment = None # 검증 시에는 보통 증강 안 함 (CLIPProcessor가 기본 처리)

    train_dataset_instance = CustomImageDataset(
        data_dir=args.train_img_dir, processor=processor, mode='train',
        augment_transform=train_augment, class_names=class_names, class_to_idx=class_to_idx
    )
    val_dataset_instance = CustomImageDataset(
        data_dir=args.train_img_dir, processor=processor, mode='val',
        augment_transform=val_augment, class_names=class_names, class_to_idx=class_to_idx
    )

    train_subset = Subset(train_dataset_instance, train_indices)
    val_subset = Subset(val_dataset_instance, val_indices)
    
    print(f'Train samples: {len(train_subset)}, Validation samples: {len(val_subset)}')


    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_skip_none)

    model = CLIPBasedModel(model_name=args.model_name, num_classes=num_classes, freeze_backbone=args.freeze_backbone).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    if args.freeze_backbone:
        optimizer = optim.AdamW(model.head.parameters(), lr=args.lr)
        print("Training only the head.")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        print("Training full model (or unfrozen backbone + head).")

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_logloss = float('inf')
    epochs_no_improve = 0 # 조기 종료 카운터

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        avg_train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scheduler)
        avg_val_loss, current_val_logloss, val_accuracy = validate_one_epoch(model, val_loader, criterion, DEVICE, num_classes)

        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} || Train Accuracy: {train_accuracy:.2f}%")
        print(f"Valid Loss: {avg_val_loss:.4f} || Valid Accuracy: {val_accuracy:.2f}% || Valid LogLoss: {current_val_logloss:.4f}")

        if current_val_logloss < best_val_logloss:
            best_val_logloss = current_val_logloss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"📦 Best model saved to {args.model_save_path} (LogLoss: {best_val_logloss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation LogLoss did not improve for {epochs_no_improve} epoch(s). Best: {best_val_logloss:.4f}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break
            
    print(f"Training finished. Best validation LogLoss: {best_val_logloss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP-based model for car classification.")
    parser.add_argument('--model_name', type=str, default=CFG['MODEL_NAME'], help="Name of the CLIP model from Hugging Face.")
    parser.add_argument('--train_img_dir', type=str, default=TRAIN_IMG_DIR, help="Path to the training images directory.")
    parser.add_argument('--img_size', type=int, default=CFG['IMG_SIZE'], help="Image size for resizing.")
    parser.add_argument('--batch_size', type=int, default=CFG['BATCH_SIZE'], help="Batch size for training and validation.")
    parser.add_argument('--epochs', type=int, default=CFG['EPOCHS'], help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=CFG['LEARNING_RATE'], help="Learning rate.")
    parser.add_argument('--seed', type=int, default=CFG['SEED'], help="Random seed.")
    parser.add_argument('--val_split', type=float, default=0.2, help="Proportion of data to use for validation.")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count()//2, help="Number of workers for DataLoader.")
    parser.add_argument('--model_save_path', type=str, default=CFG['MODEL_SAVE_PATH'], help="Path to save the best model.")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze backbone and train only the head.")
    parser.add_argument('--patience', type=int, default=CFG['PATIENCE'], help="Patience for early stopping.")


    args = parser.parse_args()
    main(args)