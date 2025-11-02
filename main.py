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
EPOCHS = 50
LR = 0.0003
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 타임스탬프 생성 (프로그램 시작 시 한 번만)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f'Timestamp: {TIMESTAMP}')
print(f'Using device: {DEVICE}')

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
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ========== 학습 함수 ==========
def train_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    losses = []
    
    for images, labels in tqdm(loader, desc='Train'):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
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
def train_fold(fold, train_df, val_df):
    print(f'\n{"="*50}')
    print(f'Fold {fold} 학습 시작')
    print(f'{"="*50}')
    
    # 데이터 로더
    train_dataset = MyDataset(train_df, train_transform)
    val_dataset = MyDataset(val_df, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 모델
    model = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=17)
    model = model.to(DEVICE)
    
    # 옵티마이저 & 스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    # 학습 루프
    best_f1 = 0
    patience_counter = 0
    patience = 10
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_f1 = validate(model, val_loader)
        
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, F1: {val_f1:.4f}')
        
        # 베스트 모델 저장
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            os.makedirs('models', exist_ok=True)
            # 파일명: fold{N}_{timestamp}_f1{score}.pth
            model_filename = f'models/fold{fold}_{TIMESTAMP}_f1{best_f1:.4f}.pth'
            torch.save(model.state_dict(), model_filename)
            print(f'✅ Best F1: {best_f1:.4f} - 저장: {model_filename}')
        else:
            patience_counter += 1
            print(f'⏳ Patience: {patience_counter}/{patience}')
        
        # Early Stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    print(f'\nFold {fold} 완료 - Best F1: {best_f1:.4f}')
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
    avg_f1 = 0
    for fold, f1, model_path in fold_info:
        model = timm.create_model('tf_efficientnetv2_m', pretrained=False, num_classes=17)
        model.load_state_dict(torch.load(model_path))
        model = model.to(DEVICE)
        model.eval()
        models.append(model)
        avg_f1 += f1
        print(f'✅ Fold {fold} (F1: {f1:.4f}) 로드')
    
    avg_f1 /= len(fold_info)
    
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
        
        # 폴드 앙상블 (평균)
        ensemble_pred = torch.stack(fold_preds).mean(dim=0)
        final_class = ensemble_pred.argmax(dim=1).item()
        all_predictions.append(final_class)
    
    return all_predictions, avg_f1

# ========== 제출 파일 생성 ==========
def create_submission(test_df, predictions, avg_f1, filename_prefix='submission'):
    """제출 파일 생성 (날짜_시간_f1score 포함)"""
    
    # 파일명: submission_{timestamp}_f1{score}.csv
    filename = f'{filename_prefix}_{TIMESTAMP}_f1{avg_f1:.4f}.csv'
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    submission.to_csv(filename, index=False)
    
    print(f'\n{"="*50}')
    print(f'제출 파일 생성: {filename}')
    print(f'{"="*50}')
    print(submission.head(10))
    print(f'\n예측 분포:')
    print(submission['label'].value_counts().sort_index())
    print(f'\n✅ 제출 파일 저장 완료!')
    
    return filename

# ========== 메인 실행 ==========
if __name__ == '__main__':
    print('='*50)
    print('이미지 분류 학습 & 추론')
    print('='*50)
    
    # ===== 1. 학습 데이터 로드 =====
    train_df = pd.read_csv('data/train.csv')
    
    print(f'학습 데이터: {len(train_df)}장')
    print(f'클래스: {train_df["label"].nunique()}개')
    
    # ===== 2. K-Fold 학습 =====
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]
        
        best_f1 = train_fold(fold, train_fold_df, val_fold_df)
        
        # 저장된 모델 파일 경로 찾기
        model_path = f'models/fold{fold}_{TIMESTAMP}_f1{best_f1:.4f}.pth'
        
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
    results_filename = f'models/fold_results_{TIMESTAMP}_avgf1{results_df["f1"].mean():.4f}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f'\n결과 저장: {results_filename}')
    
    # ===== 3. 테스트 데이터 로드 =====
    test_df = pd.read_csv('data/test.csv')
    
    # 테스트 이미지 경로 (필요시 수정)
    # test_df['image_path'] = test_df['id'].apply(lambda x: f'data/test/{x}.jpg')
    
    print(f'\n테스트 데이터: {len(test_df)}장')
    
    # ===== 4. 상위 4개 폴드 선택 =====
    results_df_sorted = results_df.sort_values('f1', ascending=False)
    best_4_folds = results_df_sorted.head(4)
    
    fold_info = [
        (row['fold'], row['f1'], row['model_path'])
        for _, row in best_4_folds.iterrows()
    ]
    
    print(f'\n선택된 폴드: {[f[0] for f in fold_info]}')
    
    # ===== 5. 앙상블 추론 =====
    predictions, avg_f1 = inference_ensemble(test_df, fold_info=fold_info, use_tta=True)
    
    # ===== 6. 제출 파일 생성 =====
    submission_filename = create_submission(test_df, predictions, avg_f1, filename_prefix='submission')
    
    print(f'\n{"="*50}')
    print('생성된 파일들')
    print(f'{"="*50}')
    print(f'📁 모델 파일:')
    for fold, f1, path in fold_info:
        print(f'  - {path}')
    print(f'\n📁 결과 파일: {results_filename}')
    print(f'📁 제출 파일: {submission_filename}')
    
    print('\n✅ 모든 작업 완료!')
```

---

## 생성되는 파일명 예시
```
models/
├── fold0_20241102_143025_f10.9423.pth
├── fold1_20241102_143025_f10.9512.pth
├── fold2_20241102_143025_f10.9387.pth
├── fold3_20241102_143025_f10.9456.pth
├── fold4_20241102_143025_f10.9401.pth
└── fold_results_20241102_143025_avgf10.9436.csv

submission_20241102_143025_f10.9450.csv
```

**파일명 구조:**
```
fold{N}_{날짜}_{시간}_f1{점수}.pth
fold_results_{날짜}_{시간}_avgf1{평균점수}.csv
submission_{날짜}_{시간}_f1{앙상블점수}.csv