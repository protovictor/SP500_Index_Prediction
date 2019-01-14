import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('sphist.csv')


def convertToDate(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date',ascending=True, inplace=True)
    return df

def addXdaysMean(df,X):
    tmp = df['Close'].rolling(X).mean()
    tmp = tmp.shift(1)
    df['mean_' + str(X)] = tmp
    return df

def addXdaysStd(df,X):
    tmp = df['Close'].rolling(X).std()
    tmp = tmp.shift(1)
    df['std_' + str(X)] = tmp
    return df

def addMeanRatio(df):
    df['meanRatio'] = df['mean_5']/df['mean_365']
    return df

def addStdRatio(df):
    df['stdRation'] = df['std_5']/df['std_365']
    return df

def removeRows(df):
    new_df = df[df["Date"] > datetime(year=1951, month=1, day=2)].copy()
    new_df.dropna(axis=0, inplace=True)
    return new_df

def transform_data(df):
    df = convertToDate(df)
    df = addXdaysMean(df, 5)
    df = addXdaysMean(df, 30)
    df = addXdaysMean(df, 365)
    df = addMeanRatio(df)
    df = addXdaysStd(df, 5)
    df = addXdaysStd(df, 30)
    df = addXdaysStd(df, 365)
    df = addStdRatio(df)
    
    df = removeRows(df)
    return df

def train_and_test(df):
    df = transform_data(df)
    train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
    test = df[df["Date"] >= datetime(year=2013, month=1, day=1)]
    
    lr = LinearRegression()
    lr.fit(train.filter(regex='^(mean)'), train['Close'])
    prediction = lr.predict(test.filter(regex='^(mean)'))
    mae = mean_absolute_error(test['Close'], prediction)
    return mae

mae = train_and_test(data)

print('mae : ' + str(mae))
