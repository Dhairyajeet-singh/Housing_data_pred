#libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


warnings.filterwarnings('ignore')

data=pd.read_csv('practice/Housing.csv') #reading data

#observing the given data
def read_data(data):
    print('shape of the data \n\n',data.shape)
    print("Name of the columns\n\n",data.columns)
    print("ndim of the data\n",data.ndim)

    print(data.head())
    print('\n')
    print(data.tail())

    for i in data.columns:
        print(i)

    print(data.info())
    print(data.describe())
    print(data.describe().transpose())
    print(data.describe(include='object'))

    print(data.isnull().sum())
    print(data.nunique())
    print(data.furnishingstatus.unique())
    print(data.furnishingstatus.value_counts())


categorical_data = data.select_dtypes(include='object').columns
# categorical_data
def category_data():
    for o in categorical_data:
        print(data[o].value_counts())
        print()
        print('---------------------------------------------------------')
        print('\n')
        sns.countplot(data=data,x=o)
        plt.show()

#data graphical representation
def data_hist():
    data.hist(figsize=(15,10), bins=50)
    plt.show()


#data outliers graphical representation
def data_outliers():
    sns.boxplot(data)
    plt.show()
    sns.boxplot(data['area'])
    plt.show()
    sns.boxplot(data['price'])
    plt.show()
#studying outliers
def outlier_study():
    print(data[data['area']>11000])
    print(len(data[data['area']>11000]))
    print(data[data['area']>11000].index)


#outliers will be removed from the dataset

def outlier_removal():
    data.drop(index=[7, 10, 56, 64, 66, 69, 125, 129, 186, 211, 403],axis=0,inplace=True)
    print(data.shape)

df=data[['area','bedrooms','bathrooms','stories','parking']]

def heatmap():
    sns.heatmap(df.corr(),annot=True)
    plt.show()

def mapping():
    data['mainroad'] = data['mainroad'].map({'yes':0, 'no':1})
    print(data['mainroad'])
    print(data.mainroad.value_counts())
    print('\n')
    data['guestroom'] = data['guestroom'].map({'yes':0, 'no':1})
    print(data['guestroom'])
    print(data.guestroom.value_counts())
    print('\n')
    data['basement'] = data['basement'].map({'yes':0, 'no':1})
    print(data['basement'])
    print(data.basement.value_counts())
    print('\n')
    data['hotwaterheating'] = data['hotwaterheating'].map({'yes':0, 'no':1})
    print(data['hotwaterheating'])
    print(data.hotwaterheating.value_counts())
    print('\n')
    data['airconditioning'] = data['airconditioning'].map({'yes':0, 'no':1})
    print(data['airconditioning'])
    print(data.airconditioning.value_counts())
    print('\n')
    data['prefarea'] = data['prefarea'].map({'yes':0, 'no':1})
    print(data['prefarea'])
    print(data.prefarea.value_counts())
    print('\n')
    data['furnishingstatus'] = data['furnishingstatus'].map({'furnished':0, 'semi-furnished':1, 'unfurnished':2})
    print(data['furnishingstatus'])
    print(data.furnishingstatus.value_counts())
    print('\n')

def model_training():
    x = data.iloc[:,1:]
    y= data['price']
    print(x)
    print(y)
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)
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

read_data(data)
category_data()
data_hist()
data_outliers()
outlier_study()
outlier_removal()
heatmap()
data_outliers()
mapping()
heatmap2()
model_training()