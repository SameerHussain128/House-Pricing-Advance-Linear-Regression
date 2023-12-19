import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv(r'D:\Data Science 6pm\Projects\Machine Learning Projects\Linear Regression\Advanced-House-Price-Prediction--master\train.csv')

## print shape of dataset with rows and columns
# It shows Missing Values of each Row
print(dataset.shape)

dataset.isnull().sum()


#________________________________________________________________________________________

df_num=dataset.select_dtypes(np.number)  # Display All Num

df_cat=dataset.select_dtypes(np.object)  # Display All Cat

#-----------------------------------------------------------------------------------------
                                           # Display All Num Col's with Missing Data
# Assuming df is your DataFrame
numerical_columns = dataset.select_dtypes(include=['int', 'float']).columns

# Check for missing values in numerical columns only
missing_values_numerical = dataset[numerical_columns].isnull().sum()

                                           # Display All Cat Col's with Missing Data
# Assuming df is your DataFrame
categorical_columns = dataset.select_dtypes(include='object').columns

# Check for missing values in categorical columns only
missing_values_categorical = dataset[categorical_columns].isnull().sum()

#________________________________________________________________________________________

dataset.fillna(dataset.median(), inplace=True)

dataset.isnull().sum()

# FILLING WITH FORWARD VALUE - PLAN A
dataset['Alley'].fillna(method='ffill', inplace=True)
dataset['MasVnrType'].fillna(method='ffill', inplace=True)
dataset['BsmtQual'].fillna(method='ffill', inplace=True)
dataset['BsmtCond'].fillna(method='ffill', inplace=True)
dataset['BsmtExposure'].fillna(method='ffill', inplace=True)
dataset['BsmtFinType1'].fillna(method='ffill', inplace=True)
dataset['BsmtFinType2'].fillna(method='ffill', inplace=True)
dataset['Electrical'].fillna(method='ffill', inplace=True)
dataset['FireplaceQu'].fillna(method='ffill', inplace=True)
dataset['GarageType'].fillna(method='ffill', inplace=True)
dataset['GarageFinish'].fillna(method='ffill', inplace=True)
dataset['GarageQual'].fillna(method='ffill', inplace=True)
dataset['GarageCond'].fillna(method='ffill', inplace=True)
dataset['PoolQC'].fillna(method='ffill', inplace=True)
dataset['Fence'].fillna(method='ffill', inplace=True)
dataset['MiscFeature'].fillna(method='ffill', inplace=True)


# CHECKING NAN MISSING VALUES 
# THIS DOWN CODE JUST FOR CHECKING
dataset['PoolQC'].isnull().sum()


# FILLING WITH MOST FREQUENT VALUE - PLAN B
dataset['Alley'].fillna(dataset['Alley'].mode()[0], inplace=True)
dataset['FireplaceQu'].fillna(dataset['Alley'].mode()[0], inplace=True)
dataset['PoolQC'].fillna(dataset['Alley'].mode()[0], inplace=True)

#----------------- UPTO HERE MISSING VALUES ARE ARE FILLER & NO MISSING LEFT ------------------------------------
 
dataset['Alley'].value_counts()
dataset['Alley'].isnull().sum()

#------------------------------------------------------------------------------------------------------
# Assuming df is your DataFrame and 'categorical_column' is the name of your categorical column
#dataset2 = pd.get_dummies(dataset, columns=['Alley'], prefix=['category'])
#dataset2 = pd.get_dummies(dataset, columns=['BsmtCond'], prefix=['category'])
#------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame and 'categorical_column' is the name of your categorical column
le = LabelEncoder()
dataset['MSZoning'] = le.fit_transform(dataset['MSZoning'])
dataset['Street'] = le.fit_transform(dataset['Street'])
dataset['Alley'] = le.fit_transform(dataset['Alley'])
dataset['LotShape'] = le.fit_transform(dataset['LotShape'])
dataset.LandContour = le.fit_transform(dataset.LandContour)
dataset.Utilities = le.fit_transform(dataset.Utilities)
dataset.LotConfig = le.fit_transform(dataset.LotConfig)
dataset.Neighborhood = le.fit_transform(dataset.Neighborhood)
dataset.Condition1 = le.fit_transform(dataset.Condition1)
dataset.Condition2 = le.fit_transform(dataset.Condition2)
dataset.BldgType = le.fit_transform(dataset.BldgType)
dataset.HouseStyle = le.fit_transform(dataset.HouseStyle)
dataset.RoofStyle = le.fit_transform(dataset.RoofStyle)
dataset.RoofMatl = le.fit_transform(dataset.RoofMatl)
dataset.Exterior1st = le.fit_transform(dataset.Exterior1st)
dataset.Exterior2nd = le.fit_transform(dataset.Exterior2nd)
dataset.MasVnrType = le.fit_transform(dataset.MasVnrType)
dataset.ExterQual = le.fit_transform(dataset.ExterQual)
dataset.ExterCond = le.fit_transform(dataset.ExterCond)
dataset.Foundation = le.fit_transform(dataset.Foundation)
dataset.BsmtQual = le.fit_transform(dataset.BsmtQual)
dataset.BsmtCond = le.fit_transform(dataset.BsmtCond)
dataset.BsmtExposure = le.fit_transform(dataset.BsmtExposure)
dataset.BsmtFinType1 = le.fit_transform(dataset.BsmtFinType1)
dataset.BsmtFinType2 = le.fit_transform(dataset.BsmtFinType2)
dataset.Heating = le.fit_transform(dataset.Heating)
dataset.HeatingQC = le.fit_transform(dataset.HeatingQC)
dataset.CentralAir = le.fit_transform(dataset.CentralAir)
dataset.Electrical = le.fit_transform(dataset.Electrical)
dataset.KitchenQual = le.fit_transform(dataset.KitchenQual)
dataset.Functional = le.fit_transform(dataset.Functional)
dataset.FireplaceQu = le.fit_transform(dataset.FireplaceQu)
dataset.GarageType = le.fit_transform(dataset.GarageType)
dataset.GarageFinish = le.fit_transform(dataset.GarageFinish)
dataset.GarageQual = le.fit_transform(dataset.GarageQual)
dataset.GarageCond = le.fit_transform(dataset.GarageCond)
dataset.PavedDrive = le.fit_transform(dataset.PavedDrive)
dataset.PoolQC = le.fit_transform(dataset.PoolQC)
dataset.Fence = le.fit_transform(dataset.Fence)
dataset.MiscFeature = le.fit_transform(dataset.MiscFeature)
dataset.SaleType = le.fit_transform(dataset.SaleType)
dataset.SaleCondition = le.fit_transform(dataset.SaleCondition)
dataset.LandSlope = le.fit_transform(dataset.LandSlope)

dataset

# Independent and Dependent Variables

X = dataset.drop(['Id','SalePrice'],axis=1)
y = dataset[['SalePrice']]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

X_train.shape, X_test.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Create a linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)
y_pred

### Evaluate the Model:

# Evaluate the model using metrics such as mean squared error and R-squared
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

### Inspect Coefficients and Intercept: 

# Coefficients and Intercept
coefficients = model.coef_
intercept = model.intercept_

print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

# The coefficients represent the weights assigned to each feature, and the intercept is the value of the target variable when all features are zero.

rmse = np.sqrt(mse)
rmse

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)
mae

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(algorithm="brute",n_neighbors=7,weights="uniform")
reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = reg.predict(X_test)
y_pred

# Evaluate the model using metrics such as mean squared error and R-squared
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Training the RF model on the Training set
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = reg.predict(X_test)
y_pred

# Evaluate the model using metrics such as mean squared error and R-squared
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Training the K-NN model on the Training set
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = reg.predict(X_test)
y_pred

# Evaluate the model using metrics such as mean squared error and R-squared
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')





