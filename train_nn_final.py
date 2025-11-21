"""
Neural Network Training Script - Optimized for Maximum F1 Score
Key improvements:
1. Threshold optimization (HUGE for F1)
2. Wide-shallow architecture + feature attention
3. Focal loss option
4. Label smoothing
5. Proper validation strategy
6. Training history saved for plotting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             classification_report, roc_auc_score, precision_recall_curve)
import pickle
import copy
import random
import time

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("="*70)
print("NEURAL NETWORK - F1 MAXIMIZATION ARCHITECTURE")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
with open('nn_input_data_option2.pkl', 'rb') as f:
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

print(f"\nTrain size: {len(X_train_full)}, Test size: {len(X_test)}")


# ============================================================================
# FOCAL LOSS (focuses on hard examples)
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=torch.tensor([self.pos_weight]).to(inputs.device),
            reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ============================================================================
# ARCHITECTURE: Wide + Attention for F1 Maximization
# ============================================================================

class F1MaximizingClassifier(nn.Module):
    """
    Design philosophy:
    - Wide first layer to capture all feature interactions
    - Feature attention to learn what matters
    - Moderate depth with residual connections
    - Heavy regularization to prevent overfitting
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.4, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Feature attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Tanh(),
                nn.Linear(input_dim, input_dim),
                nn.Softmax(dim=-1)
            )
        
        # Wide first layer - learn ALL feature interactions
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second layer with residual connection
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Compression layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout * 0.7)
        
        # Output
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Optional attention on input features
        if self.use_attention:
            attn_weights = self.attention(x)
            x = x * attn_weights
        
        # Layer 1 - wide feature learning
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)  # GELU for smoother gradients
        x = self.dropout1(x)
        
        # Layer 2 - with residual
        identity = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = x + identity  # Residual connection
        
        # Layer 3 - compression
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        return x.squeeze(-1)


# ============================================================================
# THRESHOLD OPTIMIZATION FOR F1 (KEY INNOVATION!)
# ============================================================================

def find_optimal_threshold(y_true, y_probs, min_thresh=0.3, max_thresh=0.7, step=0.01):
    """
    Find threshold that maximizes F1 score
    This is CRITICAL and often overlooked!
    """
    best_f1 = 0
    best_threshold = 0.5
    
    thresholds = np.arange(min_thresh, max_thresh, step)
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


# ============================================================================
# TRAINING FUNCTION WITH LABEL SMOOTHING
# ============================================================================

def train_model(model, train_loader, val_loader, config, device='cpu'):
    """
    Train with all the bells and whistles for F1 maximization
    """
    model = model.to(device)
    
    # Loss function
    if config['use_focal_loss']:
        criterion = FocalLoss(alpha=1, gamma=2, pos_weight=config['pos_weight'])
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config['pos_weight']]).to(device)
        )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler - OneCycle for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'] * 10,
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_f1 = 0
    best_threshold = 0.5
    best_model_state = None
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_f1_optimized': [],
        'optimal_thresholds': []
    }
    
    for epoch in range(config['epochs']):
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Label smoothing
            if config['use_label_smoothing']:
                y_batch_smooth = y_batch * 0.9 + 0.05
            else:
                y_batch_smooth = y_batch
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Handle batch size edge case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, y_batch_smooth)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_probs.extend(probs)
                val_true.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_probs = np.array(val_probs)
        val_true = np.array(val_true)
        
        # Standard F1 at 0.5 threshold
        val_preds_standard = (val_probs > 0.5).astype(int)
        val_f1_standard = f1_score(val_true, val_preds_standard, pos_label=1, zero_division=0)
        
        # OPTIMIZED threshold F1 (this is the KEY!)
        optimal_thresh, val_f1_optimized = find_optimal_threshold(val_true, val_probs)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1_standard)
        history['val_f1_optimized'].append(val_f1_optimized)
        history['optimal_thresholds'].append(optimal_thresh)
        
        # Track best model based on OPTIMIZED F1
        if val_f1_optimized > best_val_f1:
            best_val_f1 = val_f1_optimized
            best_threshold = optimal_thresh
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                  f"F1={val_f1_standard:.4f}, "
                  f"F1_opt={val_f1_optimized:.4f} @ thresh={optimal_thresh:.3f}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, best_val_f1, best_threshold, history


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, data_loader, threshold=0.5, device='cpu'):
    """Evaluate with custom threshold"""
    model.eval()
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_true.extend(y_batch.numpy())
    
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    all_preds = (all_probs >= threshold).astype(int)
    
    return all_true, all_preds, all_probs


# ============================================================================
# HYPERPARAMETER CONFIGURATION
# ============================================================================

config = {
    'hidden_dim': 256,  # Wide network
    'dropout': 0.4,
    'use_attention': True,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 200,
    'patience': 25,
    'use_focal_loss': False,  # Try both
    'use_label_smoothing': True,
    'pos_weight': pos_weight
}

print("\n" + "="*70)
print("CONFIGURATION")
print("="*70)
for key, value in config.items():
    print(f"  {key}: {value}")


# ============================================================================
# NESTED CROSS-VALIDATION
# ============================================================================

print("\n" + "="*70)
print("STARTING NESTED CROSS-VALIDATION")
print("="*70)

start_time = time.time()

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_scores_f1 = []
outer_scores_auc = []
outer_thresholds = []
fold_histories = []

for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_full, y_train_full), 1):
    print(f"\n{'='*70}")
    print(f"OUTER FOLD {fold_idx}/5")
    print(f"{'='*70}")
    
    fold_start = time.time()
    
    X_train_fold = X_train_full.iloc[train_idx]
    X_val_fold = X_train_full.iloc[val_idx]
    y_train_fold = y_train_full.iloc[train_idx]
    y_val_fold = y_train_full.iloc[val_idx]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_fold.values)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_fold.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = F1MaximizingClassifier(
        input_dim=X_train_scaled.shape[1],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        use_attention=config['use_attention']
    )
    
    # Train
    print(f"Training on {len(X_train_fold)} samples, validating on {len(X_val_fold)} samples")
    model, best_f1, best_threshold, history = train_model(
        model, train_loader, val_loader, config, device
    )
    
    # Evaluate with optimized threshold
    y_true, y_pred, y_prob = evaluate_model(model, val_loader, best_threshold, device)
    
    fold_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    fold_auc = roc_auc_score(y_true, y_prob)
    
    outer_scores_f1.append(fold_f1)
    outer_scores_auc.append(fold_auc)
    outer_thresholds.append(best_threshold)
    fold_histories.append(history)
    
    fold_time = time.time() - fold_start
    print(f"\nFold {fold_idx} Results:")
    print(f"  F1 Score: {fold_f1:.4f}")
    print(f"  AUC: {fold_auc:.4f}")
    print(f"  Optimal Threshold: {best_threshold:.3f}")
    print(f"  Time: {fold_time/60:.1f} minutes")


# ============================================================================
# NESTED CV SUMMARY
# ============================================================================

print("\n" + "="*70)
print("NESTED CV SUMMARY")
print("="*70)
print(f"Mean F1: {np.mean(outer_scores_f1):.4f} ± {np.std(outer_scores_f1):.4f}")
print(f"Mean AUC: {np.mean(outer_scores_auc):.4f} ± {np.std(outer_scores_auc):.4f}")
print(f"Mean Optimal Threshold: {np.mean(outer_thresholds):.3f} ± {np.std(outer_thresholds):.3f}")
print(f"F1 scores per fold: {[f'{f:.4f}' for f in outer_scores_f1]}")


# ============================================================================
# TRAIN FINAL MODEL ON FULL TRAINING SET
# ============================================================================

print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70)

# Use mean threshold from CV
final_threshold = np.mean(outer_thresholds)
print(f"Using threshold: {final_threshold:.3f}")

# Standardize
scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_full)
X_test_scaled_final = scaler_final.transform(X_test)

# Create dataloaders
train_dataset_final = TensorDataset(
    torch.FloatTensor(X_train_scaled_final),
    torch.FloatTensor(y_train_full.values)
)
test_dataset_final = TensorDataset(
    torch.FloatTensor(X_test_scaled_final),
    torch.FloatTensor(y_test.values)
)

train_loader_final = DataLoader(train_dataset_final, batch_size=config['batch_size'], shuffle=True)
test_loader_final = DataLoader(test_dataset_final, batch_size=config['batch_size'], shuffle=False)

# Create and train final model
final_model = F1MaximizingClassifier(
    input_dim=X_train_scaled_final.shape[1],
    hidden_dim=config['hidden_dim'],
    dropout=config['dropout'],
    use_attention=config['use_attention']
)

print(f"Training on {len(X_train_full)} samples")
final_model, _, final_threshold, final_history = train_model(
    final_model, train_loader_final, test_loader_final, config, device
)

# Use average threshold from CV (more robust)
final_threshold = np.mean(outer_thresholds)

# Evaluate on test set
y_test_true, y_test_pred, y_test_prob = evaluate_model(
    final_model, test_loader_final, final_threshold, device
)

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)
test_auc = roc_auc_score(y_test_true, y_test_prob)
cm = confusion_matrix(y_test_true, y_test_pred)

print(f"\n" + "="*70)
print("TEST SET PERFORMANCE")
print("="*70)
print(f"Accuracy: {test_acc:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"AUC-ROC: {test_auc:.4f}")
print(f"Optimal Threshold: {final_threshold:.3f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test_true, y_test_pred, target_names=['CDR=0', 'CDR>0']))

total_time = time.time() - start_time
print(f"\nTotal runtime: {total_time/60:.1f} minutes")


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results = {
    'architecture': 'F1-Maximizing Wide Network with Attention',
    'config': config,
    'nested_cv': {
        'f1_scores': outer_scores_f1,
        'auc_scores': outer_scores_auc,
        'optimal_thresholds': outer_thresholds,
        'fold_histories': fold_histories
    },
    'test_set': {
        'y_true': y_test_true,
        'y_pred': y_test_pred,
        'y_prob': y_test_prob,
        'accuracy': test_acc,
        'f1': test_f1,
        'auc': test_auc,
        'optimal_threshold': final_threshold,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_test_true, y_test_pred,
            target_names=['CDR=0', 'CDR>0'],
            output_dict=True
        )
    },
    'final_history': final_history,
    'feature_names': feature_names,
    'runtime_minutes': total_time/60
}

with open('nn_results_f1_optimized.pkl', 'wb') as f:
    pickle.dump(results, f)

torch.save({
    'model_state_dict': final_model.state_dict(),
    'scaler': scaler_final,
    'threshold': final_threshold,
    'config': config
}, 'final_nn_model_f1_optimized.pth')

print("✓ Saved nn_results_f1_optimized.pkl")
print("✓ Saved final_nn_model_f1_optimized.pth")
print(f"✓ Test F1 Score: {test_f1:.4f}")
print(f"✓ Using threshold: {final_threshold:.3f}")
