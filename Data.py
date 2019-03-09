import numpy as np
import pandas as pd
from ANNClass import ANNClass
import matplotlib.pyplot as plt

class Data:

    #data prep
    data = pd.read_csv("Dataset/defaultCCs.csv")

    def one_hot_encode(y):
        values = np.unique(y)
        return np.array([values == i for i in y])

    y = np.asarray(data["default payment next month"])
    x = np.asarray(data.iloc[:, 1])
    sex = np.asarray(data.iloc[:, 2])
    sex = one_hot_encode(sex).T
    age_cross_bal = x* np.asarray(data.iloc[:, 5])
    EDUCATION = np.asarray(data.iloc[:, 3])
    EDUCATION = one_hot_encode(EDUCATION).T
    MARRIAGE = np.asarray(data.iloc[:, 4])
    MARRIAGE = one_hot_encode(MARRIAGE).T
    cross_ed_mar = []
    for i in range(0, len(EDUCATION)):
        for k in range(0, len(MARRIAGE)):
            cross_ed_mar.append(EDUCATION[i] * MARRIAGE[k])
    cross_ed_mar = np.asarray(cross_ed_mar)
    cross_ed_bal = []
    for i in range(0, len(EDUCATION)):
        cross_ed_bal.append(EDUCATION[i] * x)
    cross_ed_bal = np.asarray(cross_ed_bal)
    invX = 1/x
    x = np.vstack((sex, EDUCATION, MARRIAGE,cross_ed_mar,cross_ed_bal,age_cross_bal,invX, x))
    for i in range(5, 24):
        x = np.vstack((np.asarray(data.iloc[:, i]), x))
    x = x- np.mean(x, axis=0)
    x / np.max(x, axis=0)
    ndata = y
    ndata = np.vstack((x, ndata))
    data = ndata.T

    #the targets of the Data should be located at the last several columns of the matrix
    #dimY = Positive Integer - columns of Y,
    #nHiddenLayers = Positive Integer - Number  of hidden Layers,
    #nNeurons = Positive Integer Tuple - number of Neurons at each respective layer,
    #activation_function = List of Strings - at each respective layer,
    #eta = Number Between 0 and 1 - learning rate for gradient,
    #batchSize = Positive Integer -  Number of Observations in each pass of Training ,
    #if Nesterov Momentum is desired then set mu to be a value greater than 0 and less than 1,
    #if RMS prop is desired then set gamma to be a value greater than 0 and less than 1,
    #if regularization is desired set lambda1/lambda2 to a value greater than zero,
    #epochs = Positive Integer - number of iter of training
    #classification = Boolean - whether or not this is a Classification problem,
    #shuffle_data = Boolean  - wheter or not te data should be shuffled

    newFit = ANNClass(data, dimY=1, nHiddenLayers = 4, nNeurons =  [55]* 4, activation_function =["tanh", "sigmoid","TanH","Sigmoid"],
                         eta = 3e-5, batchSize =1000 , mu = .9,gamma = .999,
                         lambda1 = 0, lambda2=0, epochs = 500, classification = True ,shuffle_data=True)