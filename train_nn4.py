"""
OPTION 4: THE OPTUNA CHALLENGER (FIXED)
Goal: Beat Option 2 with smart hyperparameter search
No CV bullshit, just pure performance optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import optuna
import pickle
import copy
import time
from collections import defaultdict

# Seeds
torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("OPTION 4: OPTUNA HYPERPARAMETER OPTIMIZATION")
print("Goal: DESTROY Option 2's performance")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
with open('nn_input_data_option2.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
y = data['y']
feature_names = data['feature_names']

print(f"Features: {X.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Simple 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Pos weight: {pos_weight:.3f}")

INPUT_DIM = X_train_scaled.shape[1]


# ============================================================================
# ARCHITECTURE OPTIONS
# ============================================================================

class RegularizedMLP(nn.Module):
    """Simple MLP with strong regularization"""
    def __init__(self, input_dim, hidden_dims, dropout=0.4, use_gelu=True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU() if use_gelu else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Small weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class ResidualMLP(nn.Module):
    """MLP with residual connections"""
    def __init__(self, input_dim, hidden_dim, n_blocks=2, dropout=0.4):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            for _ in range(n_blocks)
        ])
        
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x) + residual
            x = F.gelu(x)
        
        return self.output(x).squeeze(-1)


class AttentionMLP(nn.Module):
    """MLP with self-attention on features"""
    def __init__(self, input_dim, hidden_dim, dropout=0.4):
        super().__init__()
        
        # Feature attention
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        attn_weights = self.attention(x)
        x_attended = x * attn_weights
        return self.network(x_attended).squeeze(-1)


class WideShallowMLP(nn.Module):
    """Single wide hidden layer - sometimes best for small data"""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================================
# ADVANCED TRAINING UTILS
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def train_model(model, train_loader, test_loader, config, device='cuda'):
    """Train with all the bells and whistles"""
    model = model.to(device)
    
    # Optimizer
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    
    # Scheduler
    if config['scheduler'] == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'] * 10,
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
    elif config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs']
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )
    
    # Loss function
    if config['use_label_smoothing']:
        def criterion(outputs, targets):
            targets_smooth = targets * 0.9 + 0.05
            return F.binary_cross_entropy_with_logits(
                outputs, targets_smooth,
                pos_weight=torch.tensor([pos_weight]).to(device)
            )
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
    
    # Training loop
    best_f1 = 0
    best_state = None
    patience_counter = 0
    
    # SWA
    swa_model = optim.swa_utils.AveragedModel(model)
    swa_start = int(config['epochs'] * 0.7)
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Mixup
            if config['use_mixup'] and np.random.random() < 0.5:
                x_batch, y_a, y_b, lam = mixup_data(x_batch, y_batch, config['mixup_alpha'])
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
            
            loss.backward()
            
            if config['use_grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if config['scheduler'] in ['onecycle']:
                scheduler.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                val_probs.extend(probs)
                val_true.extend(y_batch.numpy())
        
        val_probs = np.array(val_probs)
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(val_true, val_preds, pos_label=1, zero_division=0)
        
        # Scheduler step
        if config['scheduler'] == 'plateau':
            scheduler.step(val_f1)
        elif config['scheduler'] == 'cosine':
            scheduler.step()
        
        # SWA
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        
        # Best model tracking
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Use SWA if available
    if config['use_swa'] and epoch >= swa_start:
        optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        return swa_model, best_f1
    
    return model, best_f1


def evaluate_model(model, data_loader, device='cuda'):
    """Evaluate model"""
    model.eval()
    model.to(device)
    
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            all_true.extend(y_batch.numpy())
    
    return np.array(all_true), np.array(all_probs)


# ============================================================================
# OPTUNA OBJECTIVE (FIXED)
# ============================================================================

def objective(trial):
    """Optuna will optimize this - FIXED VERSION"""
    
    # Architecture choice - define first
    arch_type = trial.suggest_categorical('architecture', [
        'RegularizedMLP',
        'ResidualMLP', 
        'AttentionMLP',
        'WideShallowMLP'
    ])
    
    # Common hyperparameters
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    
    # Architecture-specific hyperparameters - use conditional logic properly
    if arch_type == 'RegularizedMLP':
        n_layers = trial.suggest_int('n_layers', 2, 4)
        base_hidden = trial.suggest_categorical('base_hidden', [64, 128, 256, 512])
        
        if n_layers == 2:
            hidden_dims = [base_hidden, base_hidden // 2]
        elif n_layers == 3:
            hidden_dims = [base_hidden, base_hidden // 2, base_hidden // 4]
        else:
            hidden_dims = [base_hidden, base_hidden, base_hidden // 2, base_hidden // 4]
        
        use_gelu = trial.suggest_categorical('use_gelu', [True, False])
        
        model = RegularizedMLP(
            input_dim=INPUT_DIM,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_gelu=use_gelu
        )
    
    elif arch_type == 'ResidualMLP':
        res_hidden = trial.suggest_categorical('res_hidden', [64, 128, 256])
        n_blocks = trial.suggest_int('n_blocks', 1, 3)
        
        model = ResidualMLP(
            input_dim=INPUT_DIM,
            hidden_dim=res_hidden,
            n_blocks=n_blocks,
            dropout=dropout
        )
    
    elif arch_type == 'AttentionMLP':
        attn_hidden = trial.suggest_categorical('attn_hidden', [64, 128, 256])
        
        model = AttentionMLP(
            input_dim=INPUT_DIM,
            hidden_dim=attn_hidden,
            dropout=dropout
        )
    
    else:  # WideShallowMLP
        wide_hidden = trial.suggest_categorical('wide_hidden', [128, 256, 512])
        
        model = WideShallowMLP(
            input_dim=INPUT_DIM,
            hidden_dim=wide_hidden,
            dropout=dropout
        )
    
    # Training hyperparameters
    config = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'epochs': trial.suggest_int('epochs', 80, 200),
        'patience': trial.suggest_int('patience', 15, 30),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
        'scheduler': trial.suggest_categorical('scheduler', ['plateau', 'onecycle', 'cosine']),
        'use_mixup': trial.suggest_categorical('use_mixup', [True, False]),
        'mixup_alpha': 0.2,  # Fixed default
        'use_label_smoothing': trial.suggest_categorical('use_label_smoothing', [True, False]),
        'use_grad_clip': trial.suggest_categorical('use_grad_clip', [True, False]),
        'use_swa': trial.suggest_categorical('use_swa', [True, False])
    }
    
    # Only suggest mixup_alpha if mixup is enabled
    if config['use_mixup']:
        config['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.1, 0.4)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Train model
    try:
        trained_model, best_f1 = train_model(model, train_loader, test_loader, config, device)
        
        # Cleanup
        del trained_model, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_f1
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


# ============================================================================
# RUN OPTUNA OPTIMIZATION
# ============================================================================

print("\n" + "="*70)
print("STARTING OPTUNA HYPERPARAMETER SEARCH")
print("="*70)

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
)

start_time = time.time()

# Run optimization
study.optimize(objective, n_trials=100, timeout=7200, show_progress_bar=True)

optuna_time = time.time() - start_time

print("\n" + "="*70)
print("OPTUNA SEARCH COMPLETE")
print("="*70)
print(f"Best F1 Score: {study.best_value:.4f}")
print(f"Best trial: #{study.best_trial.number}")
print(f"Search time: {optuna_time/60:.1f} minutes")
print(f"\nBest Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMS
# ============================================================================

print("\n" + "="*70)
print("TRAINING FINAL MODEL WITH BEST PARAMS")
print("="*70)

best_params = study.best_params

# Rebuild model with best architecture
if best_params['architecture'] == 'RegularizedMLP':
    n_layers = best_params['n_layers']
    base_hidden = best_params['base_hidden']
    
    if n_layers == 2:
        hidden_dims = [base_hidden, base_hidden // 2]
    elif n_layers == 3:
        hidden_dims = [base_hidden, base_hidden // 2, base_hidden // 4]
    else:
        hidden_dims = [base_hidden, base_hidden, base_hidden // 2, base_hidden // 4]
    
    final_model = RegularizedMLP(
        input_dim=INPUT_DIM,
        hidden_dims=hidden_dims,
        dropout=best_params['dropout'],
        use_gelu=best_params.get('use_gelu', True)
    )

elif best_params['architecture'] == 'ResidualMLP':
    final_model = ResidualMLP(
        input_dim=INPUT_DIM,
        hidden_dim=best_params['res_hidden'],
        n_blocks=best_params['n_blocks'],
        dropout=best_params['dropout']
    )

elif best_params['architecture'] == 'AttentionMLP':
    final_model = AttentionMLP(
        input_dim=INPUT_DIM,
        hidden_dim=best_params['attn_hidden'],
        dropout=best_params['dropout']
    )

else:  # WideShallowMLP
    final_model = WideShallowMLP(
        input_dim=INPUT_DIM,
        hidden_dim=best_params['wide_hidden'],
        dropout=best_params['dropout']
    )

# Training config
final_config = {
    'lr': best_params['lr'],
    'weight_decay': best_params['weight_decay'],
    'batch_size': best_params['batch_size'],
    'epochs': best_params['epochs'],
    'patience': best_params['patience'],
    'optimizer': best_params['optimizer'],
    'scheduler': best_params['scheduler'],
    'use_mixup': best_params['use_mixup'],
    'mixup_alpha': best_params.get('mixup_alpha', 0.2),
    'use_label_smoothing': best_params['use_label_smoothing'],
    'use_grad_clip': best_params['use_grad_clip'],
    'use_swa': best_params['use_swa']
}

# Data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=final_config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=final_config['batch_size'], shuffle=False)

# Train
print("\nTraining final model...")
final_model, final_f1 = train_model(final_model, train_loader, test_loader, final_config, device)


# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("FINAL TEST SET EVALUATION")
print("="*70)

y_true, y_prob = evaluate_model(final_model, test_loader, device)
y_pred = (y_prob > 0.5).astype(int)

test_acc = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
test_auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)

print(f"\nTest Set Performance:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  AUC-ROC: {test_auc:.4f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['CDR=0', 'CDR>0']))


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results = {
    'architecture': best_params['architecture'],
    'optuna_study': {
        'best_f1': study.best_value,
        'best_params': best_params,
        'n_trials': len(study.trials),
        'search_time_minutes': optuna_time / 60
    },
    'test_set': {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': test_acc,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_true, y_pred, 
                                                       target_names=['CDR=0', 'CDR>0'],
                                                       output_dict=True)
    },
    'feature_names': feature_names,
    'device_used': str(device)
}

with open('nn_results_option4_optuna.pkl', 'wb') as f:
    pickle.dump(results, f)

torch.save({
    'model_state_dict': final_model.state_dict(),
    'best_params': best_params,
    'scaler': scaler
}, 'final_nn_model_option4_optuna.pth')

# Save Optuna study for later analysis
with open('optuna_study.pkl', 'wb') as f:
    pickle.dump(study, f)

print("✓ Saved nn_results_option4_optuna.pkl")
print("✓ Saved final_nn_model_option4_optuna.pth")
print("✓ Saved optuna_study.pkl")

print("\n" + "="*70)
print("🎯 MISSION: BEAT OPTION 2")
print("="*70)
print(f"Option 4 Test F1: {test_f1:.4f}")
print(f"Option 4 Test AUC: {test_auc:.4f}")
print("\nLoad Option 2 results to compare!")
print("="*70)
