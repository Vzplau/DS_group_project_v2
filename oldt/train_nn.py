"""
Neural Network Training Script - Simple Train/Val/Test Split
Standard deep learning approach (faster, production-style)
Run with: python train_nn_simple.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import pickle
import copy
from itertools import product
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("NEURAL NETWORK - Simple Train/Val/Test Split")
print("="*70)

# --- Load data ---
print("\nLoading data...")
with open('nn_input_data_simple.pkl', 'rb') as f:
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


# --- Three-way split: Train (60%), Val (20%), Test (20%) ---
print("\n" + "="*70)
print("SPLITTING DATA")
print("="*70)

# First split: separate test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate train and val (80/20 of remaining)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

print(f"Train set: {len(X_train)} samples (CDR=0: {sum(y_train==0)}, CDR>0: {sum(y_train==1)})")
print(f"Val set:   {len(X_val)} samples (CDR=0: {sum(y_val==0)}, CDR>0: {sum(y_val==1)})")
print(f"Test set:  {len(X_test)} samples (CDR=0: {sum(y_test==0)}, CDR>0: {sum(y_test==1)})")


# --- Standardize ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# --- Neural Network Architecture ---
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout=0.3):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


# --- Training function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs=150, patience=20, device='cpu', verbose=True):
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
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val F1: {val_f1:.3f}, Val AUC: {val_auc:.3f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
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
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
    
    return np.array(all_true), np.array(all_preds), np.array(all_probs)


# --- Hyperparameter grid ---
param_grid = {
    'hidden_dim1': [64],
    'hidden_dim2': [32],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64],
    'weight_decay': [0, 0.0001, 0.001]
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# --- Hyperparameter Search on Validation Set ---
print("\n" + "="*70)
print("HYPERPARAMETER SEARCH")
print("="*70)

# Generate all combinations
from itertools import product
keys = list(param_grid.keys())
values = list(param_grid.values())
all_combinations = [dict(zip(keys, v)) for v in product(*values)]

print(f"Testing {len(all_combinations)} hyperparameter combinations...")

best_val_f1 = 0
best_params = None
best_model_state = None
all_results = []

start_time = time.time()

for idx, params in enumerate(all_combinations, 1):
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # Create model
    model = BinaryClassifier(
        input_dim=X_train_scaled.shape[1],
        hidden_dim1=params['hidden_dim1'],
        hidden_dim2=params['hidden_dim2'],
        dropout=params['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                           weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Train
    print(f"\n[{idx}/{len(all_combinations)}] Testing params: {params}")
    model, val_f1, history = train_model(model, train_loader, val_loader, criterion, 
                                         optimizer, scheduler, epochs=150, patience=20, 
                                         device=device, verbose=False)
    
    print(f"  → Val F1: {val_f1:.3f}")
    
    all_results.append({
        'params': params,
        'val_f1': val_f1,
        'history': history
    })
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_params = params
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"  ★ New best F1: {best_val_f1:.3f}")

search_time = time.time() - start_time

print("\n" + "="*70)
print("HYPERPARAMETER SEARCH RESULTS")
print("="*70)
print(f"Best validation F1: {best_val_f1:.3f}")
print(f"Best hyperparameters: {best_params}")
print(f"Search time: {search_time/60:.1f} minutes")


# --- Train final model on train+val with best hyperparameters ---
print("\n" + "="*70)
print("TRAINING FINAL MODEL (Train + Val)")
print("="*70)

# Combine train and val for final training
X_train_full_scaled = scaler.fit_transform(X_train_val)
X_test_scaled_final = scaler.transform(X_test)

train_full_dataset = TensorDataset(
    torch.FloatTensor(X_train_full_scaled),
    torch.FloatTensor(y_train_val.values)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_scaled_final),
    torch.FloatTensor(y_test.values)
)

train_full_loader = DataLoader(train_full_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

# Create final model
final_model = BinaryClassifier(
    input_dim=X_train_full_scaled.shape[1],
    hidden_dim1=best_params['hidden_dim1'],
    hidden_dim2=best_params['hidden_dim2'],
    dropout=best_params['dropout']
).to(device)

criterion_final = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
optimizer_final = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'],
                              weight_decay=best_params['weight_decay'])
scheduler_final = optim.lr_scheduler.ReduceLROnPlateau(optimizer_final, mode='max', factor=0.5, patience=10)

print(f"Training with best params: {best_params}")
final_model, _, final_history = train_model(final_model, train_full_loader, test_loader,
                                             criterion_final, optimizer_final, scheduler_final,
                                             epochs=150, patience=20, device=device, verbose=True)


# --- Evaluate on test set ---
print("\n" + "="*70)
print("TEST SET EVALUATION")
print("="*70)

y_test_true, y_test_pred, y_test_prob = evaluate_model(final_model, test_loader, device)

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)
test_auc = roc_auc_score(y_test_true, y_test_prob)

print(f"\nTest Set Performance:")
print(f"  Accuracy: {test_acc:.3f}")
print(f"  F1 Score: {test_f1:.3f}")
print(f"  AUC-ROC: {test_auc:.3f}")

print(f"\nClassification Report:")
print(classification_report(y_test_true, y_test_pred, target_names=['CDR=0', 'CDR>0']))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test_true, y_test_pred))

total_time = time.time() - start_time
print(f"\nTotal runtime: {total_time/60:.1f} minutes")


# --- Save results ---
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results = {
    'hyperparameter_search': {
        'all_results': all_results,
        'best_params': best_params,
        'best_val_f1': best_val_f1
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
    'final_history': final_history,
    'feature_names': feature_names,
    'split_sizes': {
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test)
    },
    'runtime_minutes': total_time/60
}

with open('nn_results_simple.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save model
torch.save(final_model.state_dict(), 'final_nn_model_simple.pth')

print("✓ Saved nn_results_simple.pkl")
print("✓ Saved final_nn_model_simple.pth")
print("\nDownload these files and load them in your notebook!")
