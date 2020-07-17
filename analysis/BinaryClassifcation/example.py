import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import threading
from typing import List

from utils import train_test_split, standardize, to_categorical, normalize
from utils import mean_squared_error, accuracy_score
from BinaryClassifcation import host,client_1,client_2

class cli_1(threading.Thread):
    def __init__(self,feature_bins):
        threading.Thread.__init__(self)
        self.feature_bins = feature_bins
    def run(self):
        data1 = pd.read_csv('../dataset/2013_party1.csv', usecols=[1, 2, 3, 4, 5, 7, 8, 9,12])
        time1 = np.array(data1[["date", "store_nbr", "item_nbr", "family", "perishable", "city", "state"]])
        temp1 = np.atleast_2d(data1["is_hot"].values).T
        X1 = np.insert(time1, 0, values=1, axis=1)  # Insert bias term
        y1 = temp1[:, 0]
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)

        data2 = pd.read_csv('../dataset/2013_party2.csv', usecols=[1, 2, 3, 4, 5, 7, 8, 9,12])
        time2 = np.array(data2[["date", "store_nbr", "item_nbr", "family", "perishable", "city", "state"]])
        temp2 = np.atleast_2d(data2["is_hot"].values).T
        X2 = np.insert(time2, 0, values=1, axis=1)  # Insert bias term
        y2 = temp2[:, 0]  # Temperature. Reduce to one-dim
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3)

        c_1 = client_1.XGBoost()
        c_1.fit(X_train1, y_train1, self.feature_bins)

        X_test=np.append(X_test1,X_test2,axis=0)
        y_test=np.append(y_test1,y_test2,axis=0)
        y_pred = c_1.predict(X_test)
        cmap = plt.get_cmap('viridis')
        mse = mean_squared_error(y_test, y_pred)
        num=0
        num2=0
        for i in range(len(y_test)):
            if y_pred[i][0]==1 and y_test[i]==1:
                num+=1
            if y_pred[i][0]==0 and y_test[i]==0:
                num2+=1
        print(num)
        print(num2)
        print("Mean Squared Error:", mse)

        # Plot the results
        m1 = plt.scatter(range(len(y_test)), y_test, color=cmap(0.5), s=10)
        m2 = plt.scatter(range(len(y_test)), y_pred, color='black', s=10)
        plt.suptitle("Regression Tree")
        plt.xlabel('item')
        plt.ylabel('is_hot')
        plt.legend((m1, m2), ("Test data", "Prediction"), loc='lower right')
        plt.show()
        print(len(y_test))
        print(len(y_pred.reshape(-1)-y_test))
        m = plt.scatter(range(len(y_test)), y_pred.reshape(-1)-y_test, color=cmap(0.5), s=10)
        plt.title("MSE: %.2f" % mse, fontsize=10)
        plt.xlabel('item')
        plt.ylabel('is_hot')
        plt.legend([m], ("Error"), loc='lower right')
        plt.show()

class cli_2(threading.Thread):
    def __init__(self,feature_bins):
        threading.Thread.__init__(self)
        self.feature_bins = feature_bins
    def run(self):

        data2 = pd.read_csv('../dataset/2013_party2.csv', usecols=[1, 2, 3, 4, 5, 7, 8, 9,12])
        time2 = np.array(data2[["date", "store_nbr", "item_nbr", "family", "perishable", "city", "state"]])
        temp2 = np.atleast_2d(data2["is_hot"].values).T
        X2 = np.insert(time2, 0, values=1, axis=1)  # Insert bias term
        y2 = temp2[:, 0]  # Temperature. Reduce to one-dim
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3)

        c_1 = client_2.XGBoost()
        c_1.fit(X_train2, y_train2, self.feature_bins)

class h_1(threading.Thread):
    def __init__(self,feature_bins):
        threading.Thread.__init__(self)
        self.feature_bins = feature_bins
    def run(self):
        c_1 = host.XGBoost()
        c_1.fit(self.feature_bins)

def main():
    print ("-- XGBoost --")

    # Load temperature data
    data1 = pd.read_csv('../dataset/2013_party1.csv', usecols=[1, 2, 3, 4,5,7,8,9,12])
    time1 = np.array(data1[["date","store_nbr","item_nbr","family","perishable","city","state"]])
    temp1 = np.atleast_2d(data1["is_hot"].values).T
    X1 = np.insert(time1, 0, values=1, axis=1)   # Insert bias term
    y1 = temp1[:, 0]                          # Temperature. Reduce to one-dim

    data2 = pd.read_csv('../dataset/2013_party2.csv', usecols=[1, 2, 3, 4, 5, 7, 8, 9,12])
    time2 = np.array(data2[["date", "store_nbr", "item_nbr", "family", "perishable", "city", "state"]])
    temp2 = np.atleast_2d(data2["is_hot"].values).T
    X2 = np.insert(time2, 0, values=1, axis=1)  # Insert bias term
    y2 = temp2[:, 0]  # Temperature. Reduce to one-dim

    feature_bins=[]
    n_samples, n_features = np.shape(X1)
    for feature_i in range(n_features):
        feature_values1 = np.expand_dims(X1[:, feature_i], axis=1)
        feature_values2 = np.expand_dims(X2[:, feature_i], axis=1)
        unique_values = np.unique(np.append(feature_values1,feature_values2))
        feature_bins.append(unique_values)
    #print(feature_bins)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3)
    #server = host.XGBoost()
    #c_1=client_1.XGBoost()
    #c_2=client_2.XGBoost()
    #c_1.fit(X_train1, y_train1,feature_bins)

    threads = []
    h = h_1(feature_bins)
    c1 = cli_1(feature_bins)
    c2 = cli_2(feature_bins)

    threads.append(c1)
    threads.append(c2)
    threads.append(h)
    for thread in threads:
        #thread.setDaemon(True)
        thread.start()
    for thread in threads:
        thread.join()
    '''
    y_pred1 = c_1.predict(X_test1)

    #y_pred_line = model.predict(X1)
    #print(y_test[0:5])
    # Color map
    
    cmap = plt.get_cmap('viridis')
    mse = mean_squared_error(y_test1, y_pred1)

    print ("Mean Squared Error:", mse)

    # Plot the results
    m1 = plt.scatter(range(len(y_test1)), y_test1, color=cmap(0.5), s=10)
    m2 = plt.scatter(range(len(y_test1)), y_pred1, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('item')
    plt.ylabel('is_hot')
    plt.legend((m1, m2), ("Test data", "Prediction"), loc='lower right')
    plt.show()
    '''

if __name__ == "__main__":
    main()