"""
    A simple single-hidden-layer Deep Neural Network in Pytorch with 16 nodes for predicting cancer death rates from the given dataset
    
    Steps: 
    1. PreProcess dataset(scaling,encoding and imputation)
    2. Train a neural network with 16 hidden nodes 
    3. Save the trained model 
    4. Test model using saved weights 
"""

import torch 
import numpy as np
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

#Defining the neural network 

class DNN16(nn.Module):
    def __init__(self,input_dim):
        super(DNN16,self).__init__()
        self.hidden = nn.Linear(input_dim, 16) #hidden layer with 16 nodes
        self.relu = nn.ReLU()
        self.output = nn.Linear(16,1) #regression output 
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x 
    

#Training function train_model 
#Takes in the dataset and saves model to path 

def train_model(data_path="cancer_reg-1.csv",model_path="dnn16_model.pth",epochs=100,lr=0.001):
    

    # Load dataset
    df = pd.read_csv(data_path, encoding="latin-1")

    # Drop problematic column
    df = df.drop(columns=["PctSomeCol18_24"])

    # Drop suspicious duplicate rows
    duplicate_rows = ((df["avgAnnCount"] - 1962.66768).abs() < 1e-5) & \
                     ((df["incidenceRate"] - 453.549422).abs() < 1e-5)
    df = df[~duplicate_rows]

    # Separate features and target
    X = df.drop(columns=["TARGET_deathRate", "Geography"])
    Y = df["TARGET_deathRate"]

    # One-hot encode categorical column
    X = pd.get_dummies(X, columns=["binnedInc"])

    # Train/val/test split (70/15/15)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Impute missing values
    median_imputer = SimpleImputer(strategy="median")
    cols_to_impute = ["PctEmployed16_Over", "PctPrivateCoverageAlone"]

    X_train[cols_to_impute] = median_imputer.fit_transform(X_train[cols_to_impute])
    X_val[cols_to_impute] = median_imputer.transform(X_val[cols_to_impute])
    X_test[cols_to_impute] = median_imputer.transform(X_test[cols_to_impute])
    
    #Scale features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    

    print("NaNs in X_train:", np.isnan(X_train_scaled).sum())
    print("Infs in X_train:", np.isinf(X_train_scaled).sum())
    print("NaNs in Y_train:", np.isnan(Y_train.values).sum())
    print("Infs in Y_train:", np.isinf(Y_train.values).sum())

    #Converting to tensors and using view(-1,1) to flip into column 
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Initialize model
    input_dim = X_train_tensor.shape[1]
    model = DNN16(input_dim)

    # Loss and optimizer
    #Using Mean Squared Error as Loss Function and SGD as optimizer with previously set learning rate 
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    #Training loop 
    for epoch in range(epochs):
        #Forward pass 
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, Y_train_tensor)
        
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Validation loss 
        if (epoch+1) % 10 == 0:
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, Y_val_tensor).item()
            print(f"Epoch {epoch +1}/{epochs}. Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
            
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
    }, model_path)

    print(f"Model saved to {model_path}")
    

#Testing Function test_model 

def test_model(data_path="cancer_reg-1.csv", model_path="dnn16_model.pth"):
    # Load dataset and perform pre processing 
    df = pd.read_csv(data_path, encoding="latin-1")
    df = df.drop(columns=["PctSomeCol18_24"])
    duplicate_rows = ((df["avgAnnCount"] - 1962.66768).abs() < 1e-5) & \
                     ((df["incidenceRate"] - 453.549422).abs() < 1e-5)
    df = df[~duplicate_rows]
    X = df.drop(columns=["TARGET_deathRate", "Geography"])
    Y = df["TARGET_deathRate"]
    X = pd.get_dummies(X, columns=["binnedInc"])

    # Train/test split (same as training)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    _, X_test, _, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    cols_to_impute = ["PctEmployed16_Over", "PctPrivateCoverageAlone"]
    median_imputer = SimpleImputer(strategy="median")
    X_train[cols_to_impute] = median_imputer.fit_transform(X_train[cols_to_impute])
    X_test[cols_to_impute] = median_imputer.transform(X_test[cols_to_impute])

    # Load model and scaler
    checkpoint = torch.load(model_path,weights_only=False)
    scaler = checkpoint["scaler"]
    input_dim = X_train.shape[1]
    model = DNN16(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

    # Evaluate
    with torch.no_grad():
        y_test_pred = model(X_test_tensor)
        test_mse = mean_squared_error(Y_test_tensor.numpy(), y_test_pred.numpy())
        test_r2 = r2_score(Y_test_tensor.numpy(), y_test_pred.numpy())

    print(f"Test MSE: {test_mse:.2f}, Test R2: {test_r2:.2f}")


# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    train_model()
    test_model()