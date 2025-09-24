import pandas as pd 
df = pd.read_csv('HW1_Soham/cancer_reg-1.csv',encoding='latin-1')
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



