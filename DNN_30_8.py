"""
DNN_30_8.py

Two-hidden-layer neural network:
- First hidden layer: 30 neurons
- Second hidden layer: 8 neurons
- Output: 1 (regression)

Author: Soham
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# Model definition
# -------------------------
class DNN_30_8(nn.Module):
    def __init__(self, input_dim):
        super(DNN_30_8, self).__init__()
        # Layer sizes: input -> 30 -> 8 -> 1
        self.fc1 = nn.Linear(input_dim, 30)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(30, 8)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

# -------------------------
# Utility: load and preprocess
# -------------------------
def load_and_preprocess(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")

    # Drop bad column (many missing values)
    if "PctSomeCol18_24" in df.columns:
        df = df.drop(columns=["PctSomeCol18_24"])

    # Drop placeholder duplicate rows
    duplicate_rows = ((df["avgAnnCount"] - 1962.66768).abs() < 1e-5) & \
                     ((df["incidenceRate"] - 453.549422).abs() < 1e-5)
    df = df[~duplicate_rows]

    # Features / label
    X = df.drop(columns=["TARGET_deathRate", "Geography"])
    Y = df["TARGET_deathRate"].copy()

    # One-hot encode binnedInc if present
    if "binnedInc" in X.columns:
        X = pd.get_dummies(X, columns=["binnedInc"])

    return X, Y

# -------------------------
# Train function
# -------------------------
def train_model(data_path="cancer_reg-1.csv",
                model_path="dnn30_8_model.pth",
                epochs=100,
                batch_size=64,
                lr=0.001):
    """
    Train the DNN-30-8 model and save the trained weights and scaler params.
    """

    # Load data
    X, Y = load_and_preprocess(data_path)

    # Train/val/test split (70/15/15)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Impute missing values in selected cols
    cols_to_impute = [c for c in ["PctEmployed16_Over", "PctPrivateCoverageAlone"] if c in X_train.columns]
    if cols_to_impute:
        imputer = SimpleImputer(strategy="median")
        X_train[cols_to_impute] = imputer.fit_transform(X_train[cols_to_impute])
        X_val[cols_to_impute] = imputer.transform(X_val[cols_to_impute])
        X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

    # Align columns for val/test
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val.values, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                              batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = DNN_30_8(input_dim=X_train_t.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, Y_val_t).item()

        train_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}/{epochs}. Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

    # Save checkpoint (weights + scaler params + columns)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "scaler_n_features": len(scaler.mean_),
        "input_columns": list(X_train.columns)
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

# -------------------------
# Test function
# -------------------------
def test_model(data_path="cancer_reg-1.csv", model_path="dnn30_8_model.pth"):
    """
    Load trained model and evaluate on test set.
    """
    # Reload data
    X, Y = load_and_preprocess(data_path)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu",weights_only=False)

    # Rebuild scaler
    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = checkpoint["scaler_n_features"]

    # Align features
    X = X.reindex(columns=checkpoint["input_columns"], fill_value=0)

    # Train/test split again (same random_state)
    _, X_temp, _, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    _, X_test, _, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Impute again (median from training subset)
    cols_to_impute = [c for c in ["PctEmployed16_Over", "PctPrivateCoverageAlone"] if c in X.columns]
    if cols_to_impute:
        imputer = SimpleImputer(strategy="median")
        X_train_for_imputer, _, _, _ = train_test_split(X, Y, test_size=0.3, random_state=42)
        X_train_for_imputer[cols_to_impute] = imputer.fit_transform(X_train_for_imputer[cols_to_impute])
        X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

    # Scale test set
    X_test_scaled = scaler.transform(X_test)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

    # Load model
    model = DNN_30_8(input_dim=X_test_t.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predict
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()
        y_true = Y_test_t.numpy()

    test_mse = mean_squared_error(y_true, y_pred)
    test_r2 = r2_score(y_true, y_pred)
    print(f"Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")
    return test_mse, test_r2

# -------------------------
# Run directly
# -------------------------
if __name__ == "__main__":
    train_model()
    test_model()
