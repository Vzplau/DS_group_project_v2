"""
Neural Network Training Script - Option 2 Architecture
3-layer network with BatchNorm: 128 → 64 → 32
Run with: python train_nn_option2.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import pickle
import copy
import random
from itertools import product
from collections import Counter
import time
import os

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("="*70)
print("NEURAL NETWORK - Option 2 Architecture (3-layer + BatchNorm)")
print("="*70)

# --- Load data ---
print("\nLoading data...")
with open('nn_input_data_option2.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
y = data['y']
feature_names = data['feature_names']

print(f"✓ Loaded data")
print(f"  Features shape: {X.shape}")
print(f"  Class distribution: {y.value_counts().to_dict()}")

# Calculate pos_weight
pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"  Pos weight: {pos_weight:.3f}")

# --- Train/test split ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train_full)}, Test size: {len(X_test)}")


# --- Neural Network Architecture - Option 2 ---
class BinaryClassifierOption2(nn.Module):
    """
    3-layer architecture with BatchNorm:
    Input → Dense(128) + BatchNorm + ReLU + Dropout
         → Dense(64) + BatchNorm + ReLU + Dropout
         → Dense(32) + ReLU + Dropout
         → Output(1)
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.3):
        super(BinaryClassifierOption2, self).__init__()
        
        # Layer 1: Input → 128
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: 128 → 64
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3: 64 → 32
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout * 0.67)  # Reduced dropout for final layer
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim3, 1)
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        return x


# --- Training function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs=150, patience=20, device='cpu'):
    best_val_f1 = 0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            
            # Handle batch size of 1
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_probs = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch).squeeze()
                
                # Handle batch size of 1
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_true.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_true, val_preds, pos_label=1, zero_division=0)
        val_auc = roc_auc_score(val_true, val_probs) if len(set(val_true)) > 1 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_f1)
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_f1, history


# --- Evaluation function ---
def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            
            # Handle batch size of 1
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
    
    return np.array(all_true), np.array(all_preds), np.array(all_probs)


# --- Hyperparameter grid ---
param_grid = {
    'hidden_dim1': [128],
    'hidden_dim2': [64],
    'hidden_dim3': [32],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64],
    'weight_decay': [0, 0.0001, 0.001]
}

# --- Nested CV ---
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_scores_acc = []
outer_scores_f1 = []
outer_scores_auc = []
best_params_list = []
fold_predictions = []

# Force CUDA usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    print("\n⚠ CUDA not available, using CPU")

print(f"Using device: {device}")

print("\n" + "="*70)
print("NESTED CROSS-VALIDATION (5 outer folds)")
print("="*70)

start_time = time.time()

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
    
    # Inner CV: Hyperparameter search
    best_inner_f1 = 0
    best_params = None
    
    print("\nInner CV: Searching hyperparameters...")
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Sample subset
    sampled_combinations = random.sample(all_combinations, min(20, len(all_combinations)))
    
    for param_idx, params in enumerate(sampled_combinations, 1):
        inner_f1_scores = []
        
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_scaled, y_train_fold):
            X_inner_train = X_train_scaled[inner_train_idx]
            X_inner_val = X_train_scaled[inner_val_idx]
            y_inner_train = y_train_fold.iloc[inner_train_idx].values
            y_inner_val = y_train_fold.iloc[inner_val_idx].values
            
            # Create dataloaders
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
            
            # Create model (Option 2)
            model = BinaryClassifierOption2(
                input_dim=X_train_scaled.shape[1],
                hidden_dim1=params['hidden_dim1'],
                hidden_dim2=params['hidden_dim2'],
                hidden_dim3=params['hidden_dim3'],
                dropout=params['dropout']
            ).to(device)
            
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                                   weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
            
            # Train
            model, val_f1, _ = train_model(model, train_loader, val_loader, criterion, 
                                           optimizer, scheduler, epochs=100, patience=15, 
                                           device=device)
            
            inner_f1_scores.append(val_f1)
        
        mean_inner_f1 = np.mean(inner_f1_scores)
        
        if mean_inner_f1 > best_inner_f1:
            best_inner_f1 = mean_inner_f1
            best_params = params
        
        if param_idx % 5 == 0:
            print(f"  Tested {param_idx}/{len(sampled_combinations)} combinations...")
    
    print(f"\nBest params: {best_params}")
    print(f"Best inner CV F1: {best_inner_f1:.3f}")
    best_params_list.append(best_params)
    
    # Train final model on full outer fold
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
    
    model = BinaryClassifierOption2(
        input_dim=X_train_scaled.shape[1],
        hidden_dim1=best_params['hidden_dim1'],
        hidden_dim2=best_params['hidden_dim2'],
        hidden_dim3=best_params['hidden_dim3'],
        dropout=best_params['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                           weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    print("\nTraining final model on outer fold...")
    model, _, history = train_model(model, train_loader, val_loader, criterion,
                                    optimizer, scheduler, epochs=150, patience=20,
                                    device=device)
    
    # Evaluate
    y_true, y_pred, y_prob = evaluate_model(model, val_loader, device)
    
    fold_acc = accuracy_score(y_true, y_pred)
    fold_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    fold_auc = roc_auc_score(y_true, y_prob)
    
    outer_scores_acc.append(fold_acc)
    outer_scores_f1.append(fold_f1)
    outer_scores_auc.append(fold_auc)
    
    fold_predictions.append({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    
    fold_time = time.time() - fold_start
    print(f"\nOuter Fold {fold_idx} Results:")
    print(f"  Accuracy: {fold_acc:.3f}")
    print(f"  F1: {fold_f1:.3f}")
    print(f"  AUC: {fold_auc:.3f}")
    print(f"  Time: {fold_time/60:.1f} minutes")

# --- Nested CV Summary ---
print("\n" + "="*70)
print("NESTED CV SUMMARY")
print("="*70)
print(f"Mean F1: {np.mean(outer_scores_f1):.3f} ± {np.std(outer_scores_f1):.3f}")
print(f"Mean AUC: {np.mean(outer_scores_auc):.3f} ± {np.std(outer_scores_auc):.3f}")
print(f"Mean Accuracy: {np.mean(outer_scores_acc):.3f} ± {np.std(outer_scores_acc):.3f}")

# --- Final model on test set ---
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON TEST SET")
print("="*70)

final_params = {}
for param in param_grid.keys():
    values = [p[param] for p in best_params_list]
    final_params[param] = Counter(values).most_common(1)[0][0]

print(f"Using params: {final_params}")

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

final_model = BinaryClassifierOption2(
    input_dim=X_train_scaled_final.shape[1],
    hidden_dim1=final_params['hidden_dim1'],
    hidden_dim2=final_params['hidden_dim2'],
    hidden_dim3=final_params['hidden_dim3'],
    dropout=final_params['dropout']
).to(device)

criterion_final = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
optimizer_final = optim.Adam(final_model.parameters(), lr=final_params['learning_rate'],
                              weight_decay=final_params['weight_decay'])
scheduler_final = optim.lr_scheduler.ReduceLROnPlateau(optimizer_final, mode='max', factor=0.5, patience=10)

final_model, _, final_history = train_model(final_model, train_loader_final, test_loader_final,
                                             criterion_final, optimizer_final, scheduler_final,
                                             epochs=150, patience=20, device=device)

y_test_true, y_test_pred, y_test_prob = evaluate_model(final_model, test_loader_final, device)

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)
test_auc = roc_auc_score(y_test_true, y_test_prob)

print(f"\nTest Set Performance:")
print(f"  Accuracy: {test_acc:.3f}")
print(f"  F1: {test_f1:.3f}")
print(f"  AUC: {test_auc:.3f}")

total_time = time.time() - start_time
print(f"\nTotal runtime: {total_time/60:.1f} minutes")

# --- Save results ---
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results = {
    'architecture': 'Option 2: 3-layer with BatchNorm (128→64→32)',
    'nested_cv': {
        'f1_scores': outer_scores_f1,
        'auc_scores': outer_scores_auc,
        'acc_scores': outer_scores_acc,
        'best_params_list': best_params_list,
        'fold_predictions': fold_predictions
    },
    'test_set': {
        'y_true': y_test_true,
        'y_pred': y_test_pred,
        'y_prob': y_test_prob,
        'accuracy': test_acc,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': confusion_matrix(y_test_true, y_test_pred).tolist(),
        'classification_report': classification_report(y_test_true, y_test_pred, 
                                                       target_names=['CDR=0', 'CDR>0'],
                                                       output_dict=True)
    },
    'final_params': final_params,
    'final_history': final_history,
    'feature_names': feature_names,
    'runtime_minutes': total_time/60,
    'device_used': device
}

with open('nn_results_option2.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save model
torch.save(final_model.state_dict(), 'final_nn_model_option2.pth')

print("✓ Saved nn_results_option2.pkl")
print("✓ Saved final_nn_model_option2.pth")
print(f"✓ Trained on: {device}")
print("\nDownload these files and load them in your notebook!")
