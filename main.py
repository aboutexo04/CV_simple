# main.py
# 이미지 분류 - 5-Fold 앙상블 + 제출 파일 생성 (타임스탬프 포함)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import os
from datetime import datetime

# ========== 설정 ==========
IMG_SIZE = 380
BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0003
N_FOLDS = 5
MODEL_NAME = 'tf_efficientnetv2_m'  # 모델 선택: 'tf_efficientnetv2_s'(작음), 'tf_efficientnetv2_m'(중간), 'efficientnet_b0'(매우 작음)
DROPOUT_RATE = 0.4  # Dropout 비율 (0.0 ~ 1.0) - 데이터가 적으면 0.4~0.5 권장
PATIENCE = 3  # Early stopping patience (F1 개선이 없으면 중단) - 데이터가 적으면 더 짧게
WEIGHT_DECAY = 0.01  # L2 정규화 강도 (0.001~0.1, 데이터가 적으면 증가)
LABEL_SMOOTHING = 0.1  # Label Smoothing (0.0 = 사용안함, 0.1 = 권장)
USE_MIXUP = True  # MixUp augmentation 사용 여부
MIXUP_ALPHA = 0.2  # MixUp alpha 파라미터 (작을수록 더 강함, 0.1~0.4 권장)
USE_CUTMIX = False  # CutMix augmentation 사용 여부 (MixUp과 동시 사용 가능)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 타임스탬프 변수 (메인 실행 시 초기화됨 - 멀티프로세싱 안전)
TIMESTAMP = None

# ========== 데이터셋 ==========
class MyDataset(Dataset):
    def __init__(self, df, transform, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 이미지 로드
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 증강 적용
        image = self.transform(image=image)['image']
        
        if self.is_test:
            return image
        else:
            label = row['label']
            return image, label

# ========== 증강 ==========
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=10, p=0.5),  # 이동, 확대/축소, 회전
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
    ], p=0.3),  # 블러 효과
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ========== MixUp 함수 ==========
def mixup_data(x, y, alpha=1.0):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss 계산"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ========== 학습 함수 ==========
def train_epoch(model, loader, criterion, optimizer, scheduler, use_mixup=False, mixup_alpha=0.2):
    model.train()
    losses = []
    
    for images, labels in tqdm(loader, desc='Train'):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # MixUp 적용
        if use_mixup and np.random.random() > 0.5:  # 50% 확률로 MixUp 적용
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)

def validate(model, loader):
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val'):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.numpy())
    
    f1 = f1_score(labels_list, preds_list, average='macro')
    return f1

# ========== 폴드 학습 ==========
def train_fold(fold, train_df, val_df, exp_dir):
    print(f'\n{"="*50}')
    print(f'Fold {fold} 학습 시작')
    print(f'{"="*50}')
    
    # 데이터 로더
    train_dataset = MyDataset(train_df, train_transform)
    val_dataset = MyDataset(val_df, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 모델
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=17, drop_rate=DROPOUT_RATE)
    model = model.to(DEVICE)
    
    # 옵티마이저 & 스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader)
    )
    # Label Smoothing 적용
    if LABEL_SMOOTHING > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 학습 루프
    best_f1 = 0
    best_model_state = None  # 베스트 모델 상태 저장
    patience_counter = 0
    patience = PATIENCE
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                                 use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA)
        val_f1 = validate(model, val_loader)
        
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, F1: {val_f1:.4f}')
        
        # 베스트 모델 업데이트
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()  # 베스트 모델 상태 저장
            patience_counter = 0
            print(f'✅ Best F1 업데이트: {best_f1:.4f}')
        else:
            patience_counter += 1
            print(f'⏳ Patience: {patience_counter}/{patience}')
        
        # Early Stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 폴드 학습 완료 후 최종 베스트 모델 저장 (각 폴드당 하나의 파일만)
    model_filename = f'{exp_dir}/models/fold{fold}_{TIMESTAMP}_f1{best_f1:.4f}.pth'
    torch.save(best_model_state, model_filename)
    print(f'\nFold {fold} 완료 - Best F1: {best_f1:.4f} - 저장: {model_filename}')
    
    return best_f1

# ========== TTA 예측 ==========
def predict_with_tta(model, image):
    """TTA를 사용한 예측"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original
        pred = model(image)
        predictions.append(pred)
        
        # Horizontal Flip
        pred = model(torch.flip(image, dims=[3]))
        predictions.append(pred)
        
        # Vertical Flip
        pred = model(torch.flip(image, dims=[2]))
        predictions.append(pred)
        
        # Both Flips
        pred = model(torch.flip(image, dims=[2, 3]))
        predictions.append(pred)
    
    # Average
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred

# ========== 앙상블 추론 ==========
def inference_ensemble(test_df, fold_info, use_tta=True):
    """여러 폴드 모델로 앙상블 추론
    
    Args:
        test_df: 테스트 데이터프레임
        fold_info: [(fold번호, f1점수, 파일경로), ...] 리스트
        use_tta: TTA 사용 여부
    """
    print(f'\n{"="*50}')
    print(f'추론 시작')
    print(f'모델 개수: {len(fold_info)}')
    print(f'TTA: {use_tta}')
    print(f'{"="*50}')
    
    # 테스트 데이터셋
    test_dataset = MyDataset(test_df, val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 모델 로드
    models = []
    fold_f1s = []
    avg_f1 = 0
    for fold, f1, model_path in fold_info:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=17)
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model = model.to(DEVICE)
        model.eval()
        models.append(model)
        fold_f1s.append(f1)
        avg_f1 += f1
        print(f'✅ Fold {fold} (F1: {f1:.4f}) 로드')
    
    avg_f1 /= len(fold_info)
    
    # 가중치 계산 (F1 점수에 비례)
    weights = torch.tensor(fold_f1s, dtype=torch.float32)
    weights = weights / weights.sum()  # 정규화
    print(f'\n📊 앙상블 가중치: {dict(zip([f[0] for f in fold_info], weights.tolist()))}')
    
    # 추론
    all_predictions = []
    
    for images in tqdm(test_loader, desc='Inference'):
        images = images.to(DEVICE)
        
        fold_preds = []
        for model in models:
            if use_tta:
                pred = predict_with_tta(model, images)
            else:
                with torch.no_grad():
                    pred = model(images)
            
            fold_preds.append(pred.cpu())
        
        # 폴드 앙상블 (가중 평균 - F1 점수 기반)
        fold_preds_tensor = torch.stack(fold_preds)  # [num_models, batch_size, num_classes]
        weights_expanded = weights.unsqueeze(1).unsqueeze(2)  # [num_models, 1, 1]
        ensemble_pred = (fold_preds_tensor * weights_expanded).sum(dim=0)  # 가중 합
        final_class = ensemble_pred.argmax(dim=1).item()
        all_predictions.append(final_class)
    
    return all_predictions, avg_f1

# ========== 제출 파일 생성 ==========
def create_submission(test_df, predictions, avg_f1, exp_dir, filename_prefix='submission'):
    """제출 파일 생성 (날짜_시간_f1score 포함)"""
    
    # 행 수 검증
    if len(predictions) != len(test_df):
        raise ValueError(
            f'❌ 예측 결과와 테스트 데이터 행 수가 일치하지 않습니다!\n'
            f'   테스트 데이터: {len(test_df)}행\n'
            f'   예측 결과: {len(predictions)}행'
        )
    
    # 파일명: submission_{timestamp}_f1{score}.csv
    filename = f'{exp_dir}/{filename_prefix}_{TIMESTAMP}_f1{avg_f1:.4f}.csv'
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'target': predictions
    })
    
    submission.to_csv(filename, index=False)
    
    print(f'\n{"="*50}')
    print(f'제출 파일 생성: {filename}')
    print(f'{"="*50}')
    print(submission.head(10))
    print(f'\n예측 분포:')
    print(submission['target'].value_counts().sort_index())
    print(f'\n✅ 제출 파일 저장 완료!')
    
    return filename

# ========== 메인 실행 ==========
if __name__ == '__main__':
    # 타임스탬프 생성 (프로그램 시작 시 한 번만 - 멀티프로세싱 안전)
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 실험 폴더 생성 (각 실험마다 별도 폴더)
    EXP_DIR = f'experiments/exp_{TIMESTAMP}'
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(f'{EXP_DIR}/models', exist_ok=True)
    
    print('='*50)
    print('이미지 분류 학습 & 추론')
    print('='*50)
    print(f'Timestamp: {TIMESTAMP}')
    print(f'Experiment folder: {EXP_DIR}')
    print(f'Using device: {DEVICE}')
    print('='*50)
    
    # ===== 1. 학습 데이터 로드 =====
    train_df = pd.read_csv('data/train.csv')
    
    # 데이터 검증
    if len(train_df) == 0:
        raise ValueError('❌ 학습 데이터가 비어있습니다!')
    
    # 이미지 경로 추가
    train_df['image_path'] = train_df['ID'].apply(lambda x: f'data/train/{x}')
    # target 컬럼을 label로 변경 (기존 코드 호환성)
    train_df['label'] = train_df['target']
    
    # 이미지 파일 존재 확인
    missing_images = train_df[~train_df['image_path'].apply(os.path.exists)]
    if len(missing_images) > 0:
        print(f'⚠️  경고: {len(missing_images)}개의 이미지 파일을 찾을 수 없습니다.')
        print(f'첫 5개: {missing_images["ID"].head().tolist()}')
    
    print(f'학습 데이터: {len(train_df)}장')
    print(f'클래스: {train_df["label"].nunique()}개')
    
    # ===== 2. K-Fold 학습 =====
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), start=1):
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]
        
        best_f1 = train_fold(fold, train_fold_df, val_fold_df, EXP_DIR)
        
        # 저장된 모델 파일 경로 찾기
        model_path = f'{EXP_DIR}/models/fold{fold}_{TIMESTAMP}_f1{best_f1:.4f}.pth'
        
        fold_results.append({
            'fold': fold,
            'f1': best_f1,
            'model_path': model_path
        })
    
    # 결과 출력 및 저장
    results_df = pd.DataFrame(fold_results)
    print(f'\n{"="*50}')
    print('학습 결과')
    print(f'{"="*50}')
    print(results_df[['fold', 'f1']])
    print(f'\n평균 F1: {results_df["f1"].mean():.4f}')
    print(f'최고 F1: {results_df["f1"].max():.4f}')
    
    # 결과 CSV 저장 (타임스탬프 포함)
    results_filename = f'{EXP_DIR}/fold_results_{TIMESTAMP}_avgf1{results_df["f1"].mean():.4f}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f'\n결과 저장: {results_filename}')
    
    # ===== 3. 테스트 데이터 로드 =====
    # test.csv가 없으면 sample_submission.csv 사용
    if os.path.exists('data/test.csv'):
        test_df = pd.read_csv('data/test.csv')
    else:
        test_df = pd.read_csv('data/sample_submission.csv')
        # target 컬럼 제거 (예측해야 할 값이므로)
        test_df = test_df.drop('target', axis=1)
    
    # 데이터 검증
    if len(test_df) == 0:
        raise ValueError('❌ 테스트 데이터가 비어있습니다!')
    
    # 테스트 이미지 경로 추가
    test_df['image_path'] = test_df['ID'].apply(lambda x: f'data/test/{x}')
    # ID 컬럼을 id로 변경 (기존 코드 호환성)
    test_df['id'] = test_df['ID']
    
    # 이미지 파일 존재 확인
    missing_images = test_df[~test_df['image_path'].apply(os.path.exists)]
    if len(missing_images) > 0:
        print(f'⚠️  경고: {len(missing_images)}개의 테스트 이미지 파일을 찾을 수 없습니다.')
        print(f'첫 5개: {missing_images["ID"].head().tolist()}')
    
    print(f'\n테스트 데이터: {len(test_df)}장')
    
    # ===== 4. 모든 폴드 선택 (또는 상위 N개 선택) =====
    # 모든 폴드 사용 (과적합 방지 및 다양성 확보)
    # 데이터가 적을 때는 모든 폴드를 사용하는 것이 권장됨
    USE_ALL_FOLDS = True  # True: 모든 폴드 사용, False: 상위 N개만 사용
    TOP_N_FOLDS = 4  # USE_ALL_FOLDS=False일 때 사용할 상위 폴드 개수
    
    if USE_ALL_FOLDS:
        selected_folds = results_df
        print(f'\n✅ 모든 폴드 사용: {sorted(selected_folds["fold"].tolist())}')
    else:
        results_df_sorted = results_df.sort_values('f1', ascending=False)
        selected_folds = results_df_sorted.head(TOP_N_FOLDS)
        print(f'\n✅ 상위 {TOP_N_FOLDS}개 폴드 선택: {sorted(selected_folds["fold"].tolist())}')
    
    fold_info = [
        (row['fold'], row['f1'], row['model_path'])
        for _, row in selected_folds.iterrows()
    ]
    
    print(f'선택된 폴드: {[f[0] for f in fold_info]}')
    
    # ===== 5. 앙상블 추론 =====
    predictions, avg_f1 = inference_ensemble(test_df, fold_info=fold_info, use_tta=False)
    
    # ===== 6. 제출 파일 생성 =====
    submission_filename = create_submission(test_df, predictions, avg_f1, EXP_DIR, filename_prefix='submission')
    
    print(f'\n{"="*50}')
    print('생성된 파일들')
    print(f'{"="*50}')
    print(f'📁 모델 파일:')
    for fold, f1, path in fold_info:
        print(f'  - {path}')
    print(f'\n📁 결과 파일: {results_filename}')
    print(f'📁 제출 파일: {submission_filename}')
    
    print('\n✅ 모든 작업 완료!')