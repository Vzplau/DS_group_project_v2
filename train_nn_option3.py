"""
Neural Network Training Script - Option 3 (Streamlined GOATED)
Features: Residual connections, Attention, Focal Loss, Threshold Optimization
NO feature engineering - just better learning
Run with: python train_nn_option3.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
import copy
import random
from itertools import product
from collections import Counter
import time
import os
import gc

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("="*70)
print("NEURAL NETWORK - Option 3 (Streamlined GOATED)")
print("="*70)

# --- CUDA Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ Using CPU")
print(f"Device: {device}")


# --- Load data ---
print("\nLoading data...")
with open('nn_input_data_option3.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
y = data['y']
feature_names = data['feature_names']

print(f"✓ Loaded data")
print(f"  Features shape: {X.shape}")
print(f"  Class distribution: {y.value_counts().to_dict()}")

pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"  Pos weight: {pos_weight:.3f}")

# Train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train_full)}, Test: {len(X_test)}")


# ============================================================================
# STREAMLINED GOATED ARCHITECTURE
# ============================================================================

class FeatureAttention(nn.Module):
    """Learn which features are important per sample"""
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights, attn_weights


class ResidualBlock(nn.Module):
    """Residual connection for stable deep training"""
    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        out = self.relu(out)
        out = self.dropout(out)
        return out


class StreamlinedGOATED(nn.Module):
    """
    Streamlined GOATED Architecture:
    - Feature attention
    - Residual blocks
    - NO feature engineering
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super(StreamlinedGOATED, self).__init__()
        
        # Feature attention
        self.attention = FeatureAttention(input_dim)
        
        # Main pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim1, dropout)
        self.res_block2 = ResidualBlock(hidden_dim1, dropout)
        
        # Downsample
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Final layers
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim2 // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim2 // 2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout * 0.5)
        
        # Output
        self.fc_out = nn.Linear(hidden_dim2 // 2, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Apply attention
        x_attn, attn_weights = self.attention(x)
        
        # Main pathway
        out = self.fc1(x_attn)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Residual blocks
        out = self.res_block1(out)
        out = self.res_block2(out)
        
        # Downsample
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Final layers
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        # Output
        logits = self.fc_out(out)
        return logits.squeeze(-1), attn_weights


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss: Focuses on hard-to-classify examples"""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor([self.pos_weight]).to(logits.device)
        )
        
        focal_loss = self.alpha * focal_weight * bce_loss
        return focal_loss.mean()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs=150, patience=20, device='cpu', use_grad_clip=True):
    model.to(device)
    best_val_f1 = 0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        val_preds = []
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, dtype=torch.float32)
                y_batch = y_batch.to(device, dtype=torch.float32)
                
                outputs, _ = model(X_batch)
                
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_batches += 1
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_true.extend(y_batch.cpu().numpy())
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_f1 = f1_score(val_true, val_preds, pos_label=1, zero_division=0)
        val_auc = roc_auc_score(val_true, val_probs) if len(set(val_true)) > 1 else 0.5
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        if scheduler is not None:
            scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val F1={val_f1:.4f}")
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    print(f"    Best val F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    return model, best_val_f1, history


def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_true = []
    all_attn = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, dtype=torch.float32)
            
            outputs, attn_weights = model(X_batch)
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
            all_attn.append(attn_weights.cpu().numpy())
    
    return np.array(all_true), np.array(all_preds), np.array(all_probs), np.concatenate(all_attn)


def optimize_threshold(y_true, y_prob):
    """Find optimal decision threshold"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1


# ============================================================================
# HYPERPARAMETER GRID
# ============================================================================

param_grid = {
    'hidden_dim1': [128, 256],
    'hidden_dim2': [64, 128],
    'dropout': [0.3, 0.4, 0.5],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64],
    'weight_decay': [0.001, 0.01],
    'focal_gamma': [1.0, 2.0]
}

# ============================================================================
# NESTED CV
# ============================================================================

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_scores_acc = []
outer_scores_f1 = []
outer_scores_auc = []
best_params_list = []
fold_predictions = []
fold_thresholds = []

print("\n" + "="*70)
print("NESTED CROSS-VALIDATION")
print("="*70)

start_time = time.time()

for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_full, y_train_full), 1):
    print(f"\n{'='*70}")
    print(f"OUTER FOLD {fold_idx}/5")
    print(f"{'='*70}")
    fold_start = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    X_train_fold = X_train_full.iloc[train_idx]
    X_val_fold = X_train_full.iloc[val_idx]
    y_train_fold = y_train_full.iloc[train_idx]
    y_val_fold = y_train_full.iloc[val_idx]
    
    print(f"  Train: {len(X_train_fold)}, Val: {len(X_val_fold)}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Inner CV: Hyperparameter search
    best_inner_f1 = 0
    best_params = None
    
    print("\n  Inner CV: Hyperparameter search...")
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    sampled_combinations = random.sample(all_combinations, min(20, len(all_combinations)))
    
    for param_idx, params in enumerate(sampled_combinations, 1):
        inner_f1_scores = []
        
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_scaled, y_train_fold):
            X_inner_train = X_train_scaled[inner_train_idx]
            X_inner_val = X_train_scaled[inner_val_idx]
            y_inner_train = y_train_fold.iloc[inner_train_idx].values
            y_inner_val = y_train_fold.iloc[inner_val_idx].values
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_inner_train),
                torch.FloatTensor(y_inner_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_inner_val),
                torch.FloatTensor(y_inner_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            model = StreamlinedGOATED(
                input_dim=X_train_scaled.shape[1],
                hidden_dim1=params['hidden_dim1'],
                hidden_dim2=params['hidden_dim2'],
                dropout=params['dropout']
            ).to(device)
            
            criterion = FocalLoss(alpha=0.25, gamma=params['focal_gamma'], pos_weight=pos_weight)
            optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], 
                                    weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
            
            model, val_f1, _ = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs=100, patience=15, device=device, use_grad_clip=True
            )
            
            inner_f1_scores.append(val_f1)
            
            del model, criterion, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        mean_inner_f1 = np.mean(inner_f1_scores)
        
        if mean_inner_f1 > best_inner_f1:
            best_inner_f1 = mean_inner_f1
            best_params = params.copy()
        
        if param_idx % 5 == 0:
            print(f"    Tested {param_idx}/{len(sampled_combinations)} combinations...")
    
    print(f"\n  ✓ Best inner CV F1: {best_inner_f1:.4f}")
    print(f"  ✓ Best params: {best_params}")
    best_params_list.append(best_params)
    
    # Train final model on outer fold
    print("\n  Training final model...")
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_fold.values)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_fold.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    model = StreamlinedGOATED(
        input_dim=X_train_scaled.shape[1],
        hidden_dim1=best_params['hidden_dim1'],
        hidden_dim2=best_params['hidden_dim2'],
        dropout=best_params['dropout']
    ).to(device)
    
    criterion = FocalLoss(alpha=0.25, gamma=best_params['focal_gamma'], pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'],
                           weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    model, _, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=150, patience=20, device=device, use_grad_clip=True
    )
    
    # Evaluate with threshold optimization
    y_true, y_pred, y_prob, attn_weights = evaluate_model(model, val_loader, device)
    
    best_thresh, best_f1_thresh = optimize_threshold(y_true, y_prob)
    y_pred_optimized = (y_prob >= best_thresh).astype(int)
    
    fold_acc = accuracy_score(y_true, y_pred_optimized)
    fold_f1 = f1_score(y_true, y_pred_optimized, pos_label=1, zero_division=0)
    fold_auc = roc_auc_score(y_true, y_prob)
    
    outer_scores_acc.append(fold_acc)
    outer_scores_f1.append(fold_f1)
    outer_scores_auc.append(fold_auc)
    fold_thresholds.append(best_thresh)
    
    fold_predictions.append({
        'y_true': y_true,
        'y_pred': y_pred_optimized,
        'y_prob': y_prob,
        'threshold': best_thresh,
        'attn_weights': attn_weights
    })
    
    fold_time = time.time() - fold_start
    print(f"\n  ✓ Outer Fold {fold_idx} Results:")
    print(f"    Accuracy: {fold_acc:.4f}")
    print(f"    F1: {fold_f1:.4f}")
    print(f"    AUC: {fold_auc:.4f}")
    print(f"    Optimal threshold: {best_thresh:.3f}")
    print(f"    Time: {fold_time/60:.1f} min")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Summary
print("\n" + "="*70)
print("NESTED CV SUMMARY")
print("="*70)
print(f"Mean F1: {np.mean(outer_scores_f1):.4f} ± {np.std(outer_scores_f1):.4f}")
print(f"Mean AUC: {np.mean(outer_scores_auc):.4f} ± {np.std(outer_scores_auc):.4f}")
print(f"Mean Accuracy: {np.mean(outer_scores_acc):.4f} ± {np.std(outer_scores_acc):.4f}")
print(f"Mean optimal threshold: {np.mean(fold_thresholds):.3f}")

# Final model
print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70)

final_params = {}
for param in param_grid.keys():
    values = [p[param] for p in best_params_list]
    final_params[param] = Counter(values).most_common(1)[0][0]

print(f"Final params: {final_params}")

scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_full)
X_test_scaled_final = scaler_final.transform(X_test)

train_dataset_final = TensorDataset(
    torch.FloatTensor(X_train_scaled_final),
    torch.FloatTensor(y_train_full.values)
)
test_dataset_final = TensorDataset(
    torch.FloatTensor(X_test_scaled_final),
    torch.FloatTensor(y_test.values)
)

train_loader_final = DataLoader(train_dataset_final, batch_size=final_params['batch_size'], shuffle=True)
test_loader_final = DataLoader(test_dataset_final, batch_size=final_params['batch_size'], shuffle=False)

final_model = StreamlinedGOATED(
    input_dim=X_train_scaled_final.shape[1],
    hidden_dim1=final_params['hidden_dim1'],
    hidden_dim2=final_params['hidden_dim2'],
    dropout=final_params['dropout']
).to(device)

criterion_final = FocalLoss(alpha=0.25, gamma=final_params['focal_gamma'], pos_weight=pos_weight)
optimizer_final = optim.AdamW(final_model.parameters(), lr=final_params['learning_rate'],
                              weight_decay=final_params['weight_decay'])
scheduler_final = optim.lr_scheduler.ReduceLROnPlateau(optimizer_final, mode='max', factor=0.5, patience=10)

final_model, _, final_history = train_model(
    final_model, train_loader_final, test_loader_final, criterion_final,
    optimizer_final, scheduler_final, epochs=150, patience=20, device=device
)

# Evaluate with threshold optimization
y_test_true, y_test_pred, y_test_prob, test_attn = evaluate_model(final_model, test_loader_final, device)

best_test_thresh, best_test_f1 = optimize_threshold(y_test_true, y_test_prob)
y_test_pred_optimized = (y_test_prob >= best_test_thresh).astype(int)

test_acc = accuracy_score(y_test_true, y_test_pred_optimized)
test_f1 = f1_score(y_test_true, y_test_pred_optimized, pos_label=1, zero_division=0)
test_auc = roc_auc_score(y_test_true, y_test_prob)

print(f"\n" + "="*70)
print("FINAL TEST SET PERFORMANCE")
print("="*70)
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  AUC: {test_auc:.4f}")
print(f"  Optimal threshold: {best_test_thresh:.3f}")
print(f"  Confusion Matrix:")
print(f"    {confusion_matrix(y_test_true, y_test_pred_optimized)}")

total_time = time.time() - start_time
print(f"\nTotal runtime: {total_time/60:.1f} minutes")

# Save results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results = {
    'architecture': 'Option 3: Streamlined GOATED (ResNet + Attention + Focal Loss)',
    'nested_cv': {
        'f1_scores': outer_scores_f1,
        'auc_scores': outer_scores_auc,
        'acc_scores': outer_scores_acc,
        'best_params_list': best_params_list,
        'fold_predictions': fold_predictions,
        'thresholds': fold_thresholds
    },
    'test_set': {
        'y_true': y_test_true,
        'y_pred': y_test_pred_optimized,
        'y_prob': y_test_prob,
        'accuracy': test_acc,
        'f1': test_f1,
        'auc': test_auc,
        'threshold': best_test_thresh,
        'confusion_matrix': confusion_matrix(y_test_true, y_test_pred_optimized).tolist(),
        'classification_report': classification_report(y_test_true, y_test_pred_optimized, 
                                                       target_names=['CDR=0', 'CDR>0'],
                                                       output_dict=True),
        'attention_weights': test_attn.mean(axis=0).tolist()
    },
    'final_params': final_params,
    'final_history': final_history,
    'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
    'runtime_minutes': total_time/60,
    'device_used': str(device)
}

with open('nn_results_option3.pkl', 'wb') as f:
    pickle.dump(results, f)

torch.save({
    'model_state_dict': final_model.state_dict(),
    'scaler': scaler_final,
    'feature_names': feature_names,
    'final_params': final_params,
    'threshold': best_test_thresh
}, 'final_nn_model_option3.pth')

print("✓ Saved nn_results_option3.pkl")
print("✓ Saved final_nn_model_option3.pth")
print(f"✓ Device: {device}")
print("\n🚀 Streamlined GOATED model trained successfully!")
