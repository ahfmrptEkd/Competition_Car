import os
import torch
# 프로젝트 루트 디렉토리 설정 (src의 상위 디렉토리)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 

CFG = {
    # General settings
    "SEED": 42,
    
    # CLIP model settings
    "MODEL_NAME": 'openai/clip-vit-base-patch32',
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
    "EPOCHS": 7,
    "LEARNING_RATE": 5e-5,
    "NUM_CLASSES": 396,
    "PATIENCE": 3,
    "MODEL_SAVE_PATH": os.path.join(PROJECT_ROOT, 'models/clip_vit_base_patch32.pth'),
    "SUBMISSION_PATH": os.path.join(PROJECT_ROOT, 'clip_submission.csv'),
    
    # VLM model settings
    "VLM_MODEL_ID": "llava-hf/llava-1.5-7b-hf",
    "VLM_MAX_LENGTH": 2048,
    "VLM_BATCH_SIZE": 1,
    "VLM_EPOCHS": 3,
    "VLM_LEARNING_RATE": 2e-5,
    "VLM_WEIGHT_DECAY": 0.01,
    "PROMPT_TOP_K": 3,
    "TRAIN_TOP_K_JSON_PATH": os.path.join(PROJECT_ROOT, 'outputs/train_timm_efficientnet_b4_top_5.json'),
    "TEST_TOP_K_JSON_PATH": os.path.join(PROJECT_ROOT, 'outputs/test_timm_efficientnet_b4_top_5.json'),
    
    # VLM feature extraction settings
    "VLM_FEATURES_LAYER": "final_hidden_state",
    "VLM_POOLING_METHOD": "mean",
    "TRAIN_VLM_FEATURES_PATH": os.path.join(PROJECT_ROOT, 'outputs/vlm_train_features.npz'),
    "TEST_VLM_FEATURES_PATH": os.path.join(PROJECT_ROOT, 'outputs/vlm_test_features.npz'),
    
    # Final classifier settings
    "FINAL_HIDDEN_DIM": 512,
    "FINAL_DROPOUT_RATE": 0.3,
    "FINAL_LEARNING_RATE": 1e-4,
    "FINAL_WEIGHT_DECAY": 1e-4,
    "FINAL_EPOCHS": 30,
    "FINAL_PATIENCE": 5,
    "FINAL_MODEL_SAVE_PATH": os.path.join(PROJECT_ROOT, 'models/final_classifier.pth'),
    "FINAL_SUBMISSION_PATH": os.path.join(PROJECT_ROOT, 'outputs/final_submission.csv'),
}

DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_ROOT_DIR, "test")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")