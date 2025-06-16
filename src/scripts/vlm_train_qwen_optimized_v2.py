import os
import json
import argparse
import torch
import warnings
from tqdm import tqdm
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import gc

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 프로젝트 루트를 Python path에 추가 (스크립트 직접 실행을 위해)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.config.config import CFG, TRAIN_IMG_DIR, PROJECT_ROOT
from src.utils.data.enhanced_dataset import EnhancedImageDataset

class CleanQwenVLDataset(Dataset):
    """메모리 최적화된 Qwen2.5-VL 데이터셋"""
    def __init__(self,
                 image_id_to_path_map,
                 image_id_to_label_map,
                 model_a_top_k_json_path,
                 processor,
                 max_length=512,  # 줄임
                 prompt_top_k=2,   # 줄임
                 max_image_size=448  # 이미지 크기 제한 추가
                 ):
        
        self.image_id_to_path_map = image_id_to_path_map
        self.image_id_to_label_map = image_id_to_label_map
        self.processor = processor
        self.max_length = max_length
        self.prompt_top_k = prompt_top_k
        self.max_image_size = max_image_size
        
        # JSON 로드
        with open(model_a_top_k_json_path, 'r', encoding='utf-8') as f:
            self.model_a_predictions = json.load(f)

        # 유효한 샘플 필터링
        self.valid_samples = []
        for pred_id_no_ext in self.model_a_predictions.keys():
            img_id_with_ext = pred_id_no_ext + ".jpg"
            if (img_id_with_ext in self.image_id_to_path_map and 
                img_id_with_ext in self.image_id_to_label_map):
                
                image_path = self.image_id_to_path_map[img_id_with_ext]
                true_label = self.image_id_to_label_map[img_id_with_ext]
                
                if image_path and os.path.exists(image_path) and true_label:
                    self.valid_samples.append({
                        'pred_id_no_ext': pred_id_no_ext,
                        'image_path': image_path,
                        'true_label': true_label.strip(),
                        'model_a_preds': self.model_a_predictions[pred_id_no_ext]
                    })
        
        print(f"📊 Dataset: {len(self.valid_samples)} valid samples")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def resize_image(self, image):
        """이미지 크기 제한"""
        width, height = image.size
        if max(width, height) > self.max_image_size:
            if width > height:
                new_width = self.max_image_size
                new_height = int(height * self.max_image_size / width)
            else:
                new_height = self.max_image_size
                new_width = int(width * self.max_image_size / height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        return image
    
    def __getitem__(self, idx):
        try:
            sample = self.valid_samples[idx]
            image = Image.open(sample['image_path']).convert("RGB")
            
            # 이미지 차원 일치를 위한 수정
            # 목표: 모든 이미지를 정사각형으로 만들어 일관성 확보
            target_dim = self.max_image_size

            img_w, img_h = image.size

            # 이미지가 이미 목표 크기의 정사각형인지 확인
            if img_w == target_dim and img_h == target_dim:
                image_to_process = image
            else:
                # 종횡비를 유지하면서 target_dim x target_dim 안에 맞도록 이미지 리사이즈
                aspect_ratio = img_w / img_h

                if aspect_ratio > 1:  # 가로가 더 긴 이미지
                    new_w_aspect = target_dim
                    new_h_aspect = int(target_dim / aspect_ratio)
                else:  # 세로가 더 길거나 정사각형 이미지
                    new_h_aspect = target_dim
                    new_w_aspect = int(target_dim * aspect_ratio)
                
                # 리사이즈된 크기가 최소 1픽셀이 되도록 보장
                new_w_aspect = max(1, new_w_aspect)
                new_h_aspect = max(1, new_h_aspect)

                resized_image_aspect = image.resize((new_w_aspect, new_h_aspect), Image.LANCZOS)

                # 리사이즈된 이미지를 정사각형 배경에 붙여넣기 (패딩)
                padding_color = (128, 128, 128)
                image_to_process = Image.new("RGB", (target_dim, target_dim), padding_color)

                # 리사이즈된 이미지를 중앙에 배치하기 위한 좌상단 좌표 계산
                paste_x = (target_dim - new_w_aspect) // 2
                paste_y = (target_dim - new_h_aspect) // 2

                image_to_process.paste(resized_image_aspect, (paste_x, paste_y))

            # 간단한 프롬프트 생성
            model_a_preds = sample['model_a_preds']
            if model_a_preds:
                pred_parts = []
                for pred in model_a_preds[:self.prompt_top_k]:
                    class_name = pred.get('class_name', 'unknown')
                    confidence = pred.get('confidence', 0.0)
                    pred_parts.append(f"{class_name}({confidence:.0%})")
                
                pred_str = ", ".join(pred_parts)
                prompt = (f"Analyze this car's visual features and identify the model. "
                        f"CV predictions: {pred_str}. "
                        f"Describe brand markers, body style, and design elements.")
            else:
                prompt = "Analyze this car image. Describe the brand, model, and distinctive visual characteristics."

            # 메시지 포맷
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_to_process},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 프로세싱
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = f"{text}{sample['true_label']}<|im_end|>"
            
            inputs = self.processor(
                text=[full_text],
                images=[image_to_process],
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=False
            )

            # 레이블 처리
            input_ids = inputs['input_ids'].squeeze(0)
            labels = input_ids.clone()

            # 프롬프트 마스킹
            try:
                prompt_tokens = self.processor.tokenizer(
                    text, return_tensors="pt", add_special_tokens=False,
                )["input_ids"].squeeze(0)
                if len(prompt_tokens) < len(labels):
                    labels[:len(prompt_tokens)] = -100
            except:
                pass

            result = {
                "input_ids": input_ids,
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": labels,
                "image_id": sample['pred_id_no_ext'],
            }
            
            if "pixel_values" in inputs:
                result["pixel_values"] = inputs["pixel_values"].squeeze(0)
            if "image_grid_thw" in inputs:
                result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
            
            return result

        except Exception as e:
            return None

class CleanQwenVLDataCollator:
    """깔끔한 콜레이터"""
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}
        
        max_len = max(item["input_ids"].size(0) for item in batch)
        
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_grid_thw = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            
            # 패딩
            pad_length = max_len - input_ids.size(0)
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id
                
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), -100, dtype=labels.dtype)
                ])

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_labels.append(labels)

            if "pixel_values" in item and item["pixel_values"] is not None:
                batch_pixel_values.append(item["pixel_values"])
            if "image_grid_thw" in item and item["image_grid_thw"] is not None:
                batch_image_grid_thw.append(item["image_grid_thw"])

        result = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_masks),
            "labels": torch.stack(batch_labels)
        }

        if batch_pixel_values:
            result["pixel_values"] = torch.stack(batch_pixel_values)
        if batch_image_grid_thw:
            result["image_grid_thw"] = torch.stack(batch_image_grid_thw)

        return result

# 메모리 정리를 위한 콜백
class MemoryCleanupCallback(TrainerCallback):
    """메모리 정리 콜백"""
    def on_step_end(self, args, state, control, **kwargs):
        # 50스텝마다 메모리 정리
        if state.global_step % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return control

def load_model_optimized(model_id, use_awq=False, use_4bit=False):
    """메모리 최적화된 모델 로드"""
    
    # AWQ 모델 매핑
    if use_awq:
        awq_models = {
            "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        }
        target_model_id = awq_models.get(model_id, model_id)
        if target_model_id != model_id:
            model_id = target_model_id
    
    # 모델 로드 설정
    if use_4bit and not use_awq:
        # 4-bit quantization 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # 4bit 모델 준비
        model = prepare_model_for_kbit_training(model)
        # 4bit 모델에서만 gradient checkpointing 활성화
        model.gradient_checkpointing_enable()
    else:
        # AWQ 또는 일반 모델 로드
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # bfloat16 대신 float16 사용
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )
        # AWQ 모델에서는 gradient checkpointing 비활성화

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, processor, tokenizer

def setup_lora_optimized(model, args):
    """최적화된 LoRA 설정"""
    # 학습 가능한 파라미터만 활성화
    model.requires_grad_(False)
    
    # LoRA target modules 최적화 (VLM에 맞게)
    target_modules = args.lora_target_modules.split(',')
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None,  # AWQ와의 호환성을 위해 명시적으로 None
    )

    model_with_lora = get_peft_model(model, lora_config)
    
    # 학습 가능한 파라미터 수 출력
    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    print(f"📊 Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model_with_lora

def prepare_data_maps(train_img_dir):
    """데이터 매핑 준비"""
    clip_dataset = EnhancedImageDataset(
        data_dir=train_img_dir,
        processor=None,
        mode='train',
        model_type='clip'
    )

    image_id_to_path_map = {}
    image_id_to_label_map = {}

    for i in tqdm(range(len(clip_dataset)), desc="Mapping"):
        try:
            info = clip_dataset.get_img_info(i)
            img_id = info['image_id']
            img_path = info['abs_path']
            label_text = info['label_text']
            
            if img_id and img_path and label_text and os.path.exists(img_path):
                image_id_to_path_map[img_id] = img_path
                image_id_to_label_map[img_id] = label_text
        except:
            continue
    
    return image_id_to_path_map, image_id_to_label_map
    
def create_safe_subset(image_id_to_path_map, image_id_to_label_map, subset_size):
    """서브셋 생성"""
    common_keys = set(image_id_to_path_map.keys()) & set(image_id_to_label_map.keys())
    common_keys = list(common_keys)
    
    if subset_size > 0 and subset_size < len(common_keys):
        selected_keys = common_keys[:subset_size]
        subset_path_map = {k: image_id_to_path_map[k] for k in selected_keys}
        subset_label_map = {k: image_id_to_label_map[k] for k in selected_keys}
        print(f"📊 Using subset: {len(selected_keys)} samples")
        return subset_path_map, subset_label_map
    else:
        filtered_path_map = {k: image_id_to_path_map[k] for k in common_keys}
        filtered_label_map = {k: image_id_to_label_map[k] for k in common_keys}
        print(f"📊 Using full dataset: {len(common_keys)} samples")
        return filtered_path_map, filtered_label_map

def main(args):
    print(f"🚀 Starting Qwen2.5-VL {args.model_id.split('/')[-1]} Fine-tuning (Optimized)")

    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 데이터 준비
    image_id_to_path_map, image_id_to_label_map = prepare_data_maps(TRAIN_IMG_DIR)
    if image_id_to_path_map is None:
        return
    
    image_id_to_path_map, image_id_to_label_map = create_safe_subset(
        image_id_to_path_map, image_id_to_label_map, args.subset_size
    )

    # 모델 로드
    print(f"🔧 Loading {args.model_id} (AWQ: {args.use_awq}, 4bit: {args.use_4bit})")
    model, processor, tokenizer = load_model_optimized(args.model_id, args.use_awq, args.use_4bit)
    model_with_lora = setup_lora_optimized(model, args)

    # 데이터셋
    dataset = CleanQwenVLDataset(
        image_id_to_path_map=image_id_to_path_map,
        image_id_to_label_map=image_id_to_label_map,
        model_a_top_k_json_path=args.train_top_k_json,
        processor=processor,
        max_length=args.max_length,
        prompt_top_k=args.prompt_top_k,
        max_image_size=args.max_image_size
    )

    data_collator = CleanQwenVLDataCollator(processor)

    # 학습 설정 - AWQ와 4bit에 따라 다르게 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=True,  # fp16 사용
        tf32=False,  # TF32 비활성화
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,  # 메모리 절약
        report_to=[],
        optim="adamw_torch",  # 기본 optimizer 사용
        gradient_checkpointing=args.use_4bit and not args.use_awq,  # 4bit일 때만 활성화
        max_grad_norm=1.0,  # gradient clipping
        warmup_ratio=0.03,  # warmup 추가
    )

    # 트레이너 (기본 Trainer 사용)
    trainer = Trainer(
        model=model_with_lora,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[MemoryCleanupCallback()]  # 콜백으로 메모리 정리
    )

    # 학습 실행
    print("🚀 Starting training...")
    print(f"📊 Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    try:
        trainer.train()

        final_path = os.path.join(args.output_dir, "final_checkpoint")
        trainer.save_model(final_path)
        processor.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

        print(f"✅ Training completed! Model saved to {final_path}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"📊 Peak GPU memory: {memory_used:.2f} GB")
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        
        error_path = os.path.join(args.output_dir, "error_checkpoint")
        os.makedirs(error_path, exist_ok=True)
        try:
            model_with_lora.save_pretrained(error_path)
            processor.save_pretrained(error_path)
            print(f"💾 Error checkpoint saved: {error_path}")
        except:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Memory-Optimized Qwen2.5-VL Fine-tuning")
    
    parser.add_argument('--train_top_k_json', type=str, required=True)
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--use_awq', action='store_true', help='Use AWQ quantized model')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization (not with AWQ)')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--prompt_top_k', type=int, default=2)
    parser.add_argument('--max_image_size', type=int, default=448, help='Maximum image size')
    
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target_modules', type=str, 
                       default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--subset_size', type=int, default=0)
    
    parser.add_argument('--output_dir', type=str, 
                       default=os.path.join(PROJECT_ROOT, "outputs", "qwen_optimized"))
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    main(args)