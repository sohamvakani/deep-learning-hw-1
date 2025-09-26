import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt 
import joblib
import time



#Load Data
df = pd.read_csv('HW1_Soham/cancer_reg-1.csv',encoding='latin-1')
pd.options.display.float_format = '{:.2f}'.format
#print(df.head())
#getting information about the dataframe 
# print("Info about dataframe", df.info())
# print("Shape of dataframe: ",df.shape)
#Getting minimum and maximum values of columns for step 1, question 3 
#This includes all columns with datatype number 
numerical_columns = df.select_dtypes(include='number')
min_values = numerical_columns.min(numeric_only=True)
# print()
# print("Minimum values for numerical columns:", min_values)
max_values = numerical_columns.max(numeric_only=True)
# print()
# print("Maximum values for numerical columns:", max_values)
#Checking for missing values in the dataset 
missing_values = df.isnull().sum()
# print()
# print("Missing values:")
# print(missing_values[missing_values>0])
#Drop column PctSomeCol18_24 due to 75% missing values 
df = df.drop(columns=["PctSomeCol18_24"])
count_repeated_rows = (df["avgAnnCount"] == 1962.667684).sum()
# print("Repeated placeholder data count:",count_repeated_rows)

#Drop rows with suspicious repeated values 
#avg AnnCount = 1962.66768 and incidenceRate = 453.549422 
duplicate_rows = ((df["avgAnnCount"] - 1962.66768).abs() < 1e-5) & \
                 ((df["incidenceRate"] - 453.549422).abs() < 1e-5)
df = df[~duplicate_rows]
# print(df.info())

#Seperating features (X) and label (Y) 
X = df.drop(columns=["TARGET_deathRate","Geography"])
Y = df["TARGET_deathRate"]

unique_count = X["binnedInc"].nunique()
# print("Unique count:",unique_count)
#one-hot encoding categorial column binnedInc
X = pd.get_dummies(X,columns=['binnedInc'])

#Splitting dataset into training,testing and validation 
#Splitting into train(70%) and temp(30%) 
X_train, X_temp , Y_train, Y_temp = train_test_split(X,Y,test_size=0.3, random_state=42)

#Splitting temp data into validation(15%) and test(15%)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp,Y_temp,test_size=0.5,random_state=42)
# print(X_train.shape,X_val.shape,X_test.shape)
# print(Y_train.shape,Y_val.shape,Y_test.shape)

#Imputing median values for missing values 
median_imputer = SimpleImputer(strategy='median')
#Creating list of columns to impute values for 
cols_to_impute = ['PctEmployed16_Over','PctPrivateCoverageAlone']
#Fitting the imputer on the training data and transforming it 
X_train[cols_to_impute] = median_imputer.fit_transform(X_train[cols_to_impute])

#Transform the validation and test data using the imputer fitted on the training data
X_val[cols_to_impute] = median_imputer.transform(X_val[cols_to_impute])
X_test[cols_to_impute] = median_imputer.transform(X_test[cols_to_impute])

# print("Missing values in PctEmployed16_Over after imputation:")
# print("Train:", X_train['PctEmployed16_Over'].isnull().sum())
# print("Validation:", X_val['PctEmployed16_Over'].isnull().sum())
# print("Test:", X_test['PctEmployed16_Over'].isnull().sum())

#Creating Linear regression model 
linreg = LinearRegression() 
#Training the model 
start = time.time()
linreg.fit(X_train,Y_train)
end = time.time()

print(f"Linear Regression Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
print(f"Linear Regression Training ended at   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
print(f"Linear Regression Total training time: {end - start:.4f} seconds")
Y_pred = linreg.predict(X_val)

#getting MSE and R2 value for validation set 
mse_lin_reg = mean_squared_error(Y_val,Y_pred)
r2 = r2_score(Y_val,Y_pred)
print(f'MSE for Linear Regression: {mse_lin_reg:.2f} , R^2:{r2:.2f}')

joblib.dump(linreg, "linear_regression.pkl")

#Getting R2 value for test set 
#Predict on the test set 
Y_test_predict = linreg.predict(X_test)
#Calculate R2 score on test set 
Test_r2 = r2_score(Y_test,Y_test_predict)
print()
print(f'Test R2 for Linear Regression:{Test_r2:.2f}')


def test_model(model_path, X_test, Y_test):
    """
    Load a trained model from disk and evaluate it on the test set.
    Parameters:
        model_path (str): Path to the saved model file (.pkl)
        X_test (pd.DataFrame or np.array): Test features
        Y_test (pd.Series or np.array): Test labels
    Returns:
        dict: Dictionary with predictions, MSE, and R^2 score
    """
    # Load the trained model
    model = joblib.load(model_path)
    # Make predictions
    Y_pred = model.predict(X_test)
    # Evaluate performance
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R^2: {r2:.2f}")
    return {"predictions": Y_pred, "mse": mse, "r2": r2}

#Training Neural Networks 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# DNN with 1 hidden layer of 16 nodes
dnn = MLPRegressor(hidden_layer_sizes=(16,),
                   activation='relu',
                   solver='sgd',
                   max_iter=10000,
                   random_state=42,
                   learning_rate_init=0.0001)   

# Train
start = time.time()
dnn.fit(X_train_scaled, Y_train)
end = time.time()

print(f"DNN-16 Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
print(f"DNN-16 Training ended at   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
print(f"DNN-16 Total training time: {end - start:.4f} seconds")
# Validate
Y_val_pred_dnn = dnn.predict(X_val_scaled)
mse_val = mean_squared_error(Y_val,Y_val_pred_dnn)
r2_val_dnn = r2_score(Y_val, Y_val_pred_dnn)
print(f'Validation R2 for single layer neural network: {r2_val_dnn:.2f}')
print(f"Validation MSE for single layer neural network: {mse_val:.2f}")
# Test
Y_test_pred_dnn = dnn.predict(X_test_scaled)
r2_test_dnn = r2_score(Y_test, Y_test_pred_dnn)
print(f'Test R2 for single layer neural network: {r2_test_dnn:.2f}')

loss_values_dnn = dnn.loss_curve_
epochs = range(1, len(loss_values_dnn) + 1)

# Create the plot
plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
plt.plot(epochs, loss_values_dnn, label='Training MSE Loss', color='blue', linestyle='-')

# Add titles and labels to make the plot informative
plt.title('Model Training Performance for DNN-16')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE) Loss')
plt.ylim(0,1000)
plt.legend()
plt.grid(True)
# Show the final plot
plt.show()

# DNN of two layers 
joblib.dump(dnn, "dnn_best.pkl") 

# Create DNN with 2 hidden layers: 30 nodes in first, 8 nodes in second
dnn_two_layers = MLPRegressor(hidden_layer_sizes=(30, 8),
                              activation='relu',
                              solver='sgd',
                              max_iter=3000,
                              random_state=42,
                              learning_rate_init=0.0001)

# Train the model
start = time.time()
dnn_two_layers.fit(X_train_scaled, Y_train)
end = time.time()

print(f"DNN-30-8 Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
print(f"DNN-30-8 Training ended at   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
print(f"DNN-30-8 Total training time: {end - start:.4f} seconds")


# Predict on validation set
Y_val_pred_dnn_two = dnn_two_layers.predict(X_val_scaled)
mse_val_two = mean_squared_error(Y_val, Y_val_pred_dnn_two)
r2_val_two = r2_score(Y_val, Y_val_pred_dnn_two)
print(f'Validation MSE for DNN of two layers: {mse_val_two:.2f}, R^2: {r2_val_two:.2f}')

# Predict on test set
Y_test_pred_dnn_two = dnn_two_layers.predict(X_test_scaled)
r2_test_dnn_two = r2_score(Y_test, Y_test_pred_dnn_two)
print(f'Test R^2 for DNN of two layers : {r2_test_dnn_two:.2f}')


loss_values_dnn_two = dnn_two_layers.loss_curve_
epochs = range(1, len(loss_values_dnn_two) + 1)

# Create the plot
plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
plt.plot(epochs, loss_values_dnn_two, label='Training MSE Loss', color='blue', linestyle='-')

# Add titles and labels to make the plot informative
plt.title('Model Training Performance for DNN-30-8')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE) Loss')
plt.ylim(0,1000)
plt.legend()
plt.grid(True)
# Show the final plot
plt.show()

dnn_three_layers = MLPRegressor(hidden_layer_sizes=(30,16,8),
                                activation='relu',
                                solver='sgd',
                                max_iter=5000,
                                random_state=42,
                                learning_rate_init=0.0001)

#Train the model 
start = time.time()
dnn_three_layers.fit(X_train_scaled, Y_train)
end = time.time()

print(f"DNN-30-16-8 Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
print(f"DNN-30-16-8 Training ended at   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
print(f"DNN-30-16-8 Total training time: {end - start:.4f} seconds")


#Predict on validation set 
Y_val_pred_dnn_three = dnn_three_layers.predict(X_val_scaled)
mse_val_three = mean_squared_error(Y_val, Y_val_pred_dnn_three)
r2_val_three = r2_score(Y_val,Y_val_pred_dnn_three)
print(f"Validation MSE for DNN of three layers: {mse_val_three:.2f}, R^2:{r2_val_three:.2f}")

Y_test_pred_dnn_three = dnn_three_layers.predict(X_test_scaled)
r2_test_dnn_three = r2_score(Y_test,Y_test_pred_dnn_three)
print(f'Test R^2 for DNN of three layers: {r2_test_dnn_three:.2f}')

loss_values = dnn_three_layers.loss_curve_
epochs = range(1, len(loss_values) + 1)

# Create the plot
plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
plt.plot(epochs, loss_values, label='Training MSE Loss', color='blue', linestyle='-')

# Add titles and labels to make the plot informative
plt.title('Model Training Performance for DNN-30-16-8')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE) Loss')
plt.ylim(0,1000)
plt.legend()
plt.grid(True)


# Show the final plot
plt.show()

dnn_four_layers = MLPRegressor(hidden_layer_sizes=(30,16,8,4),
                               activation='relu',
                               solver='adam',
                               learning_rate_init=0.0001,
                               momentum=0.9,
                               max_iter=10000,
                               early_stopping=True,
                               n_iter_no_change=50,
                               validation_fraction=0.15,
                               random_state=42)

# Train the model
start = time.time()
dnn_four_layers.fit(X_train_scaled, Y_train)
end = time.time()

print(f"DNN-30-16-8-4 Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
print(f"DNN-30-16-8-4 Training ended at   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
print(f"DNN-30-16-8-4 Total training time: {end - start:.4f} seconds")


# Predict on validation set
Y_val_pred_dnn_four = dnn_four_layers.predict(X_val_scaled)
mse_val_four = mean_squared_error(Y_val, Y_val_pred_dnn_four)
r2_val_four = r2_score(Y_val, Y_val_pred_dnn_four)
print(f"Validation MSE for DNN of four layers: {mse_val_four:.2f}, R^2:{r2_val_four:.2f}")

# Predict on test set
Y_test_pred_dnn_four = dnn_four_layers.predict(X_test_scaled)
r2_test_dnn_four = r2_score(Y_test, Y_test_pred_dnn_four)
print(f'Test R^2 for DNN of four layers: {r2_test_dnn_four:.2f}')


loss_values_dnn_four = dnn_four_layers.loss_curve_
epochs = range(1, len(loss_values_dnn_four) + 1)

# Create the plot
plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
plt.plot(epochs, loss_values_dnn_four, label='Training MSE Loss', color='blue', linestyle='-')

# Add titles and labels to make the plot informative
plt.title('Model Training Performance for DNN-30-16-8-4')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE) Loss')
plt.ylim(0,1000)
plt.legend()
plt.grid(True)
# Show the final plot
plt.show()


