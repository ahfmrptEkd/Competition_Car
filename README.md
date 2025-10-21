# 🚗 HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회

- 대회기간 : 2025.05 ~ 2025.06 (1 month)
- 주최/주관: 헥토(Hecto) / 데이콘
**VLM 기반 자동차 분류 시스템 - 개인 대회 프로젝트**

> 핵토 Vision AI 챌린지 상위 335% 달성: CLIP, EfficientNet+SentenceTransformer, 사전 학습된 VLM 특징 등 고급 컴퓨터 비전 및 비전-언어 모델을 구현 및 비교하여 높은 정확도의 자동차 이미지 검색 시스템 구축

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-latest-red.svg)
![Transformers](https://img.shields.io/badge/transformers-latest-yellow.svg)
## 🚀 기술 스택

* **핵심 라이브러리**: Python 3.9+, PyTorch, NumPy, Pandas
* **비전 모델**: torchvision (`EfficientNet`), `clip` 라이브러리
* **언어/VL 모델**: 🤗 Transformers (`CLIPModel`, `Qwen/Qwen-VL-Chat`, `LLaVA-1.5-7B`), Sentence Transformers
* **유틸리티**: OpenCV (`opencv-python`), Scikit-learn, tqdm, PyYAML
---


## 🎯 프로젝트 개요

* **S (Situation / 문제)**
  
  [헥토 AI 챌린지](https://dacon.io/competitions/official/236493/overview/description)는 주어진 쿼리 이미지와 시각적으로 차의 종류를 분류하는 강력한 시스템 구축을 요구했습니다. 이 과제는 서로 다른 자동차 모델, 각도, 조명 조건을 효과적으로 구별하는 높은 정밀도와 재현율을 필요로 합니다. 또한 CV 대회였으나, 본 프로젝트는 VLM의 가능성과 한계를 탐구하는 데 중점을 두고 진행되었습니다.
<br>

* **T (Task / 목표)**
  
  대회 리더보드에서 가능한 가장 높은 정확도(특히 Log Loss으로 측정)를 달성하기 위해 이미지 검색을 위한 여러 최첨단 접근 방식을 개발하고 평가하는 것이 주요 목표였습니다. 주요 목표에는 종단 간 학습(End-to-End Training)과 특징 추출(Feature Extraction) 방식 비교, 시각 정보와 텍스트 정보 결합 등이 포함되었습니다.
<br>

* **A (Action / 해결)**

  1. **다중 모드 접근법 탐색**: 여러 전략을 구현하고 비교했습니다.
     * **종단 간 CLIP**: 이미지와 자동차 모델 이름에 대한 공동 임베딩을 학습하기 위해 CLIP (Contrastive Language–Image Pre-training) 모델(`src/clip/`)을 학습시켰습니다.
     * **CV + 텍스트 융합**: 시각적 특징을 위해 EfficientNet 모델(`src/cv/training/enhanced_train.py`)을 학습시키고, `car_model_name`에서 파생된 Sentence Transformers(`src/cv/inference/enhanced_inference.py`)의 텍스트 임베딩과 결합했습니다.
     * **사전 학습된 VLM 특징**: CLIP (`src/pretrained/quick_vlm_features.py`) 및 Qwen-VL (`src/pretrained/qwen_vlm_features.py`)과 같은 강력한 사전 학습된 비전-언어 모델을 사용하여 심층 특징을 추출하고, 이 특징들로 경량 모델을 학습시켰습니다 (`src/scripts/`).

  2. **최적화 및 실험 추적**: 최적화된 학습 루프 (`src/scripts/vlm_train_optimized.py`, `src/scripts/vlm_train_qwen_optimized_v2.py`), 설정 관리 (`src/config/config.py`)을 구현했습니다.

  3. **추론 파이프라인**: 특징 추출, 유사도 검색(코사인 유사도), 상위 k개 예측 생성(`src/cv/teammate_topk_generator.py`)을 통합한 추론 스크립트(`src/clip/inference.py`, `src/cv/inference/enhanced_inference.py`)를 개발했습니다.
<br>

* **R (Result / 결과)**
  
  * **상위 35% 달성**: 구현된 전략들, 특히 사전 학습된 VLM 특징과 텍스트 데이터를 결합한 접근 방식은 공식 대회 리더보드에서 **상위 35%** 순위를 기록했습니다.
  * **비교 분석**: 다양한 접근 방식의 효과를 성공적으로 입증했으며, 특히 이 이미지 검색 작업에 대한 사전 학습된 VLM 특징 사용의 강력한 성능을 강조했습니다.
  * **모듈식 코드베이스**: 다양한 모델과 기술을 쉽게 실험할 수 있는 잘 구조화된 코드베이스를 개발했습니다.

---
<br>

## 📁 프로젝트 구조
```
├── README.md
├── requirements.txt      # 프로젝트 의존성
├── setup.py              # 기본 패키지 설정
├── src/
│   ├── clip/             # CLIP 모델 구현 (데이터셋, 모델, 학습, 추론)
│   ├── config/           # 설정 파일 (config.py)
│   ├── cv/               # 컴퓨터 비전 모델 (EfficientNet) 및 SentenceTransformer 융합
│   │   ├── inference/
│   │   └── training/
│   ├── pretrained/       # VLM 특징 추출 스크립트 (CLIP, Qwen-VL)
│   ├── scripts/          # 사전 추출된 특징을 사용하는 최적화된 학습 스크립트
│   └── utils/            # 유틸리티 함수 (시드, 로깅 등)
└── data/                 # (대회 데이터 자리 표시자)
    ├── train/
    ├── test/
    ├── train_df.csv
    └── sample_submission.csv
```

<br>

## 🛠️ 기술적 구현 (Action)

이 프로젝트는 자동차 이미지 검색 문제를 해결하기 위해 여러 경로를 탐색했습니다.

### 1. 아키텍처 및 접근 방식

* **CLIP (Contrastive Language-Image Pretraining)** (`src/clip/`):
  * **목표**: 유사한 자동차 이미지와 해당 `car_model_name` 텍스트 설명이 서로 가까워지는 공유 임베딩 공간을 학습합니다.
  * **구현**: PyTorch와 `clip` 라이브러리를 사용하여 `CLIPDataset` (`src/clip/dataset.py`)과 `CLIPModel` (`src/clip/model.py`)을 정의했습니다. 대조 손실(contrastive loss)을 최적화하는 표준 학습 루프(`src/clip/train.py`)를 구현했습니다. 추론은 쿼리 이미지와 후보 이미지/텍스트를 모두 임베딩하고 코사인 유사도를 사용하여 가장 가까운 이웃을 찾는 방식으로 이루어집니다.

* **EfficientNet + SentenceTransformer 융합** (`src/cv/`):
  * **목표**: 강력한 CNN의 시각적 특징과 의미론적 텍스트 특징을 결합합니다.
  * **구현**: 시각적 분류/특징 추출을 위해 자동차 이미지에 대해 `EfficientNet` 모델(`src/cv/training/enhanced_train.py`)을 미세 조정했습니다. `car_model_name` 텍스트를 임베딩하기 위해 사전 학습된 `SentenceTransformer` (`snunlp/KR-SBERT-V40K-klueNLI-augSTS`)를 사용했습니다. 추론 중 (`src/cv/inference/enhanced_inference.py`)에는 유사도를 계산하기 전에 시각적 임베딩과 텍스트 임베딩을 연결하거나 가중치를 부여했습니다.

* **사전 학습된 VLM 특징 추출** (`src/pretrained/`, `src/scripts/`):
  * **목표**: 광범위한 미세 조정 없이 대규모 VLM이 학습한 강력한 표현(representation)을 활용합니다.
  * **구현**:
    * Hugging Face `transformers`와 `openai/clip-vit-base-patch32`, `Qwen/Qwen-VL-Chat` 같은 모델을 사용하여 이미지 특징을 효율적으로 추출하는 스크립트(`quick_vlm_features.py`, `qwen_vlm_features.py`)를 개발했습니다. 특징은 로컬에 저장했습니다.
    * 미리 추출된 특징을 로드하고 PyTorch를 사용하여 간단한 분류 헤드(예: 선형 레이어)를 학습시키는 최적화된 학습 스크립트를 만들었습니다. 이는 종단 간 미세 조정에 비해 학습 시간과 자원 요구 사항을 크게 줄였습니다.

* **VLM Fine-tuning** (`src/scripts/vlm_train_qwen_optimized_v2.py`):
    * **목표**: 제한된 로컬 GPU 환경에서 7B 이상의 거대 VLM(Qwen2.5-VL)을 직접 파인튜닝하여, 단순 특징 추출 방식보다 더 높은 성능을 달성합니다.
    * **구현**:
        * 4-bit Quantization (양자화): transformers의 BitsAndBytesConfig를 사용해 모델을 4비트로 로드(load_in_4bit=True)하여 VRAM 사용량을 획기적으로 절감했습니다.
        * LoRA (Low-Rank Adaptation): peft 라이브러리의 LoraConfig를 적용하여, 모델 전체 파라미터가 아닌 일부 어댑터 레이어만 학습시켰습니다. (Parameter-Efficient Fine-Tuning).
        * 하이브리드 프롬프트: CV 모델(CLIP)이 생성한 Top-K 예측 결과를 VLM에 텍스트 힌트(예: "CV predictions: 싼타페(70%), 쏘렌토(20%)...")로 함께 제공하여, VLM이 더 정확한 답을 생성하도록 유도했습니다.
<br>

### 2. 주요 구현 세부 정보

* **설정 관리**: 재현성을 보장하기 위해 하이퍼파라미터, 파일 경로, 모델 설정을 관리하는 데 Python `dataclasses` (`src/config/config.py`)를 사용했습니다.
* **유틸리티**: 무작위 시드 설정, 로깅, 데이터 처리와 같은 공통 함수를 `src/utils/utils.py`에 중앙 집중화했습니다.
* **추론 및 제출**: 쿼리와 후보 임베딩 간의 코사인 유사도를 계산하고, 상위 k개 예측을 생성(`teammate_topk_generator.py`)하며, 앙상블을 통해 대회 제출 요구 사항에 따라 결과를 형식화하는 추론 파이프라인을 구현했습니다.

---
<br>

## 📊 결과 및 성능 (Result)

* **대회 순위**: [헥토 Vision AI 챌린지 리더보드](https://dacon.io/competitions/official/236493/overview/description)에서 **상위 35%** 를 달성했습니다.
* **지표**: 주요 평가 지표는 **Log Loss** 이었습니다. VLM 특징 추출 접근 방식이 비공개 리더보드에서 가장 좋은 결과를 냈습니다.
* **정성적 결과**:
  * Qwen-VL 및 CLIP과 같은 **사전 학습된 비전-언어 모델** 을 특징 추출에 활용하는 것이 처음부터 모델을 학습하거나 기존 CNN에만 의존하는 것보다 이 작업에 매우 효과적이고 계산적으로 효율적임이 입증되었습니다.
  * 시각적 특징과 함께 관련 **텍스트 정보** (`car_model_name`)를 통합하는 것이 다양한 아키텍처에서 검색 정확도를 일관되게 향상시켰습니다.
    
| 지표 | 결과 |
|------|------|
| **최종 순위** | **255/748** (상위 35%) |
| **초기 점수** | 3.5170318914 (LogLoss) |
| **최종 점수** | **0.2811784058** (LogLoss) |
| **성능 개선** | **92% 향상** |
| **1등 점수** | 0.08849 (참고) |

---
<br>

## 💡 주요 학습 내용

* **사전 학습된 VLM의 힘**: 대규모 사전 학습 모델은 이미지 검색과 같은 특정 작업에 효과적으로 전이되는 풍부한 시각적 및 의미적 표현을 포착하며, 종종 더 작은 데이터셋으로 학습된 작업별 모델보다 성능이 뛰어납니다.
* **특징 추출 대 미세 조정**: 시간/자원이 제한된 대회의 경우, 강력한 사전 학습 모델에서 특징을 추출하고 간단한 헤드를 학습시키는 것이 매우 효과적이고 효율적인 전략이 될 수 있습니다.
* **다중 모드 융합**: 시각적 유사성만으로는 충분하지 않을 수 있는 세분화된 이미지 검색의 경우, 시각적 특징과 관련 텍스트 메타데이터(예: 자동차 모델 이름)를 결합하는 것이 중요합니다.
