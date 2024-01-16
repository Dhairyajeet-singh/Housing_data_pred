import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

data= pd.read_csv('project/USA_Housing.csv')

def data_read():  #understanding the given dataset
    print(data)
    print(data.shape)
    print(data.head())
    print(data.tail())
    print(data.describe())
    print(data.describe(include='object'))
    print(data.describe().transpose())
    print(data.columns)
    print(data.info())
    print(data.describe(include='object'))
    print('\n')
    for i in data.columns:
        print('\n',i)

    print(data['Avg. Area Number of Rooms'].unique())
    print(data.nunique())
    print(data.isnull().sum())
    print(data['Address'])

def numeric_data():
    numerical_data = data.select_dtypes(include=np.number).columns
    # numerical_data
    for col in numerical_data:
        print(data[col].value_counts())
        print()
        print('---------------------------------------------------------')

#categorical data
def category_data():
    categorical_data=data.select_dtypes(include='object').columns
    for col in categorical_data:
        print(data[col].value_counts())
        print()
        print('--------------------------------------------------------------------')
        print('\n')
        print(categorical_data.nunique())

#data graphical representation
def data_hist():
    data.hist(figsize=(15,10), bins=50)
    plt.show()

def data_outliers():
    sns.boxplot(data)
    plt.show()
    plt.boxplot(data['Avg. Area Income'])
    plt.show()
    plt.boxplot(data['Price'])
    plt.show()

def outlier_study():
    print(data[(data['Avg. Area Income'] > 97000) | (data['Avg. Area Income'] < 40000)])
    print(len(data[(data['Avg. Area Income'] > 97000) | (data['Avg. Area Income'] < 40000)]))
    print(data[(data['Avg. Area Income'] > 97000) | (data['Avg. Area Income'] < 40000)].index)


def outlier_removal():
    data.drop(index=[  12,   39,  411,  428,  558,  693,  844,  962, 1096, 1271, 1459, 1597,
       1734, 1891, 2025, 2092, 2242, 2300, 2597, 2719, 3069, 3144, 3183, 3483,
       3541, 3798, 3947, 4087, 4400, 4449, 4716, 4744, 4844, 4855],axis=0, inplace=True)
    print(data)

df=data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]

def heatmap():
    sns.heatmap(df.corr(), annot=True)
    plt.show()
def mapping():
    data['address']= data['address'].map({1})
def model_training():
    x = data.iloc[:,1:]
    y= data['Price']
    print(x)
    print(y)
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)
    label_encoder = LabelEncoder()
    x_train['Address'] = label_encoder.fit_transform(x_train['Address'])
    x_test['Address'] = label_encoder.fit_transform(x_test['Address'])
    print('x_train \n',x_train)
    print('x_test \n',x_test)
    print('\n',x.shape)
    print('y_train \n',y_train)
    print('y_test \n',y_test)

    lr=LinearRegression()
    lr.fit(x_train,y_train)
    y_train_pred=lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    print('y_train2\n',y_train)
    print(len(y_train_pred))
    print('y_train_pred\n',y_train_pred)
    print('y_test\n',y_test)
    print('y_test_pred\n',y_test_pred)
    print('lr.coef \n',lr.coef_)

    def graph_plot():
        plt.figure(figsize=(10,6))#best fit line for train data
        plt.scatter(y_train, y_train_pred)
        plt.show()
        plt.figure(figsize=(10,6))
        plt.scatter(y_train, y_train_pred, color='m', label='Actual Price')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4, label='Predicted Price')
        plt.title('Best fit line on Training Data')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.show()
        plt.figure(figsize=(10,6))
        plt.scatter(y_test,y_test_pred)
        plt.show()
        plt.figure(figsize=(10,6))
        plt.scatter(y_test, y_test_pred, color='c', label='Actual Price')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='Predicted Price')
        plt.title('Best fit line on Test Data')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.show()
    graph_plot()

    def training_data():
        mean_squared_error(y_train,y_train_pred)
        mean_absolute_error(y_train,y_train_pred)
        r2_score(y_train,y_train_pred)
        def training_evaluation(actual, predicted):
            mse=mean_squared_error(actual,predicted)
            rmse=np.sqrt(mean_squared_error(actual,predicted))
            mae=mean_absolute_error(actual,predicted)
            r2=r2_score(actual,predicted)
            print('MSE:\n',mse)
            print('RMSE:\n',rmse)
            print('MAE:\n',mae)
            print('R2_score:',r2)
        print('Training Set Evaluation: ', '\n----------------------------------')
        training_evaluation(y_train, y_train_pred)
    training_data()
    def test_data():
        mean_squared_error(y_test,y_test_pred)
        mean_absolute_error(y_test,y_test_pred)
        r2_score(y_test,y_test_pred)
        def test_evaluation(actual, predicted):
            mse=mean_squared_error(actual,predicted)
            rmse=np.sqrt(mean_squared_error(actual,predicted))
            mae=mean_absolute_error(actual,predicted)
            r2=r2_score(actual,predicted)
            print('MSE:\n',mse)
            print('RMSE:\n',rmse)
            print('MAE:\n',mae)
            print('R2_score:',r2)
        print('Training Set Evaluation: ', '\n----------------------------------')
        test_evaluation(y_test, y_test_pred)
    test_data()


def heatmap2():
    plt.figure(figsize=(15,10))
    sns.heatmap(data.corr(),annot=True)
    plt.show()

data_read()
numeric_data()
category_data()
data_hist()
data_outliers()
outlier_study()
outlier_removal()
heatmap()

model_training()