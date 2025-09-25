import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

#Load Data
df = pd.read_csv('cancer_reg-1.csv',encoding='latin-1')
pd.options.display.float_format = '{:.2f}'.format
#print(df.head())
#getting information about the dataframe 
print("Info about dataframe", df.info())
print("Shape of dataframe: ",df.shape)
#Getting minimum and maximum values of columns for step 1, question 3 
#This includes all columns with datatype number 
numerical_columns = df.select_dtypes(include='number')
min_values = numerical_columns.min(numeric_only=True)
print()
print("Minimum values for numerical columns:", min_values)
max_values = numerical_columns.max(numeric_only=True)
print()
print("Maximum values for numerical columns:", max_values)
#Checking for missing values in the dataset 
missing_values = df.isnull().sum()
print()
print("Missing values:")
print(missing_values[missing_values>0])
#Drop column PctSomeCol18_24 due to 75% missing values 
df = df.drop(columns=["PctSomeCol18_24"])
count_repeated_rows = (df["avgAnnCount"] == 1962.667684).sum()
print("Repeated placeholder data count:",count_repeated_rows)

#Drop rows with suspicious repeated values 
#avg AnnCount = 1962.66768 and incidenceRate = 453.549422 
duplicate_rows = ((df["avgAnnCount"] - 1962.66768).abs() < 1e-5) & \
                 ((df["incidenceRate"] - 453.549422).abs() < 1e-5)
df = df[~duplicate_rows]
print(df.info())

#Seperating features (X) and label (Y) 
X = df.drop(columns=["TARGET_deathRate","Geography"])
Y = df["TARGET_deathRate"]

unique_count = X["binnedInc"].nunique()
print("Unique count:",unique_count)
#one-hot encoding categorial column binnedInc
X = pd.get_dummies(X,columns=['binnedInc'])

#Splitting dataset into training,testing and validation 
#Splitting into train(70%) and temp(30%) 
X_train, X_temp , Y_train, Y_temp = train_test_split(X,Y,test_size=0.3, random_state=42)

#Splitting temp data into validation(15%) and test(15%)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp,Y_temp,test_size=0.5,random_state=42)
print(X_train.shape,X_val.shape,X_test.shape)
print(Y_train.shape,Y_val.shape,Y_test.shape)

#Imputing median values for missing values 
median_imputer = SimpleImputer(strategy='median')
#Creating list of columns to impute values for 
cols_to_impute = ['PctEmployed16_Over','PctPrivateCoverageAlone']
#Fitting the imputer on the training data and transforming it 
X_train[cols_to_impute] = median_imputer.fit_transform(X_train[cols_to_impute])

#Transform the validation and test data using the imputer fitted on the training data
X_val[cols_to_impute] = median_imputer.transform(X_val[cols_to_impute])
X_test[cols_to_impute] = median_imputer.transform(X_test[cols_to_impute])

print("Missing values in PctEmployed16_Over after imputation:")
print("Train:", X_train['PctEmployed16_Over'].isnull().sum())
print("Validation:", X_val['PctEmployed16_Over'].isnull().sum())
print("Test:", X_test['PctEmployed16_Over'].isnull().sum())

#Creating Linear regression model 
linreg = LinearRegression() 
#Training the model 
linreg.fit(X_train,Y_train)
Y_pred = linreg.predict(X_val)

#getting MSE and R2 value for validation set 
mse_lin_reg = mean_squared_error(Y_val,Y_pred)
r2 = r2_score(Y_val,Y_pred)
print(f'MSE for Linear Regression: {mse_lin_reg:.2f} , R^2:{r2:.2f}')

#Getting R2 value for test set 
#Predict on the test set 
Y_test_predict = linreg.predict(X_test)
#Calculate R2 score on test set 
Test_r2 = r2_score(Y_test,Y_test_predict)
print()
print(f'Test R2 for Linear Regression:{Test_r2:.2f}')


