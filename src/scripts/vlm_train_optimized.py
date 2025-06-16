# vlm_train_optimized.py - 극한의 메모리 최적화 버전

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gc

# 프로젝트 루트를 Python path에 추가 (스크립트 직접 실행을 위해)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 프로젝트 모듈 임포트
from src.config.config import CFG, DEVICE, TRAIN_IMG_DIR, PROJECT_ROOT
from src.utils.data.enhanced_dataset import EnhancedImageDataset
from src.utils.data.vlm_dataset import LlavaFineTuningDataset
from src.utils.utils import seed_everything

def clear_memory():
    """GPU 메모리 정리"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_optimized_model(model_name="llava-hf/llava-1.5-7b-hf"):
    """극한으로 최적화된 모델 로드"""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    
    # 더 공격적인 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # bfloat16 대신 float16
    )
    
    # CPU offloading 활성화
    device_map = {
        "vision_tower": 0,  # GPU
        "multi_modal_projector": 0,  # GPU
        "language_model.model.embed_tokens": 0,  # GPU
        "language_model.model.layers": 0,  # GPU에 할당
        "language_model.model.norm": 0,  # GPU
        "language_model.lm_head": 0,  # GPU로 변경
    }
    
    print("🔄 Loading model with aggressive optimization...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder="offload",  # 오프로드 폴더
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Gradient checkpointing 활성화
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # 더 작은 LoRA 설정
    lora_config = LoraConfig(
        r=8,  # 16에서 8로 감소
        lora_alpha=16,  # 32에서 16으로 감소
        target_modules=["q_proj", "v_proj"],  # 타겟 모듈 줄이기
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor

def train_with_minimal_memory(args):
    """최소 메모리로 학습"""
    seed_everything(args.seed)
    clear_memory()
    
    # 1. 데이터 준비 (기존 코드 재사용)
    print("📊 Preparing data...")
    clip_dataset = EnhancedImageDataset(
        data_dir=TRAIN_IMG_DIR,
        processor=None,
        mode='train',
        model_type='clip'
    )
    
    # 이미지 ID 매핑 생성
    image_id_to_path_map = {}
    image_id_to_label_map = {}
    
    for i in range(len(clip_dataset)):
        info = clip_dataset.get_img_info(i)
        image_id_to_path_map[info['image_id']] = info['abs_path']
        image_id_to_label_map[info['image_id']] = info['label_text']
    
    # 2. 모델 로드
    model, processor = get_optimized_model(args.model_name)
    clear_memory()
    
    # 3. 데이터셋 생성 (작은 서브셋만 사용)
    if args.subset_size > 0:
        # 데이터 서브셋만 사용
        subset_ids = list(image_id_to_path_map.keys())[:args.subset_size]
        subset_path_map = {k: image_id_to_path_map[k] for k in subset_ids}
        subset_label_map = {k: image_id_to_label_map[k] for k in subset_ids}
    else:
        subset_path_map = image_id_to_path_map
        subset_label_map = image_id_to_label_map
    
    vlm_dataset = LlavaFineTuningDataset(
        image_id_to_path_map=subset_path_map,
        image_id_to_label_map=subset_label_map,
        model_a_top_k_json_path=args.train_top_k_json,
        processor=processor,
        max_length=args.max_length,
        prompt_top_k=args.prompt_top_k
    )
    
    # 4. 최적화된 학습 설정
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        
        # 극한의 메모리 최적화
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        
        # Mixed precision
        fp16=True,
        fp16_full_eval=False,
        
        # 옵티마이저 설정
        optim="adamw_bnb_8bit",  # 8-bit optimizer
        learning_rate=args.learning_rate,
        
        # 저장 설정
        save_strategy="epoch",
        save_total_limit=1,
        
        # 메모리 최적화
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
        
        # DeepSpeed (선택사항)
        # deepspeed="ds_config.json",  # DeepSpeed 설정 파일
        
        # 기타
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Custom collate function
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}
        
        # 패딩 대신 개별 처리
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }
    
    # 5. Trainer 생성 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=vlm_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )
    
    print("🚀 Starting optimized training...")
    try:
        # 학습 전 메모리 정리
        clear_memory()
        
        # 학습
        trainer.train()
        
        # 저장
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        
        print("✅ Training completed successfully!")
        
    except torch.cuda.OutOfMemoryError:
        print("❌ Still OOM! Try these options:")
        print("1. Reduce --max_length (current: {})".format(args.max_length))
        print("2. Increase --grad_accum_steps (current: {})".format(args.grad_accum_steps))
        print("3. Use --subset_size to train on fewer samples")
        print("4. Use a smaller model with --model_name")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_top_k_json', type=str, required=True)
    parser.add_argument('--model_name', type=str, 
                       default="llava-hf/llava-1.5-7b-hf",
                       help="Model to use (try smaller ones if OOM)")
    parser.add_argument('--max_length', type=int, default=768,  # 줄임
                       help='Maximum sequence length')
    parser.add_argument('--prompt_top_k', type=int, default=3)
    parser.add_argument('--grad_accum_steps', type=int, default=16)  # 늘림
    parser.add_argument('--epochs', type=int, default=1)  # 줄임
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--subset_size', type=int, default=0,
                       help='Use only N samples (0 = use all)')
    parser.add_argument('--output_dir', type=str, 
                       default='outputs/vlm_optimized')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train_with_minimal_memory(args)
