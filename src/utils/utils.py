import os
import random
import numpy as np
import pandas as pd # multiclass_log_loss를 위해 추가
import torch
from sklearn.metrics import log_loss

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {}
    
    return torch.utils.data.dataloader.default_collate(batch)

def calculate_log_loss(true_labels_list, pred_probs_array, class_labels_list):
    """
    간단화된 log_loss 계산 함수.
    true_labels_list: 정답 레이블 리스트
    pred_probs_array: 예측 확률 배열
    class_labels_list: 클래스 레이블 리스트
    """
    # 확률값 클리핑 (log(0) 방지)
    clipped_probs = np.clip(pred_probs_array, 1e-15, 1 - 1e-15)
    return log_loss(true_labels_list, clipped_probs, labels=class_labels_list)
