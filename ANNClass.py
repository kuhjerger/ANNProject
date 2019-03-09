import numpy as np
import matplotlib.pyplot as plt

class ANNClass:
    def __init__(self, data, dimY = 1, nHiddenLayers = 4, nNeurons = (8,8,8,8),  activation_function =["relu"] * 4,
                 eta = 1e-4, batchSize = 10 , mu = 0, gamma = 1,lambda1 = 0, lambda2 = 0, epochs = 20,
                 classification = False, shuffle_data = False):
        '''the targets of the Data should be located at the last several columns of the matrix
        dimY = Positive Integer - columns of Y,
        nHiddenLayers = Positive Integer - Number  of hidden Layers,
        nNeurons = Positive Integer Tuple - number of Neurons at each respective layer,
        activation_function = List of Strings - at each respective layer,
        eta = Number Between 0 and 1 - learning rate for gradient,
        batchSize = Positive Integer -  Number of Observations in each pass of Training ,
        if Nesterov Momentum is desired then set mu to be a value greater than 0 and less than 1,
        if RMS prop is desired then set gamma to be a value greater than 0 and less than 1,
        if regularization is desired set lambda1/lambda2 to a value greater than zero,
        epochs = Positive Integer - number of iter of training
         classification = Boolean - whether or not this is a Classification problem,
         shuffle_data = Boolean  - wheter or not te data should be shuffled '''



        X,Y, K = self.dataPrep(data, dimY,classification )
        W, b = self.buildLayers(nHiddenLayers, nNeurons, X.shape[1], Y.shape[1])

        for i in range(len(activation_function)):
            activation_function[i] = activation_function[i].lower()

        act_fun = ["relu"]*(len(W)-1)

        if len(activation_function) == len(act_fun):
            for i in range(len(activation_function)):
                if activation_function[i] != act_fun[i]:
                    act_fun[i] = activation_function[i]

        print("Used Activation Functions ", act_fun)

        self.fitData(X,Y,W,b,eta,batchSize,mu, gamma,lambda1,lambda2,act_fun,epochs,classification,shuffle_data)


    def dataPrep(self,data, dimY,classification):
        K = 0
        if classification == False:
            if dimY ==1:
                y = data[:,-1]
                N = len(y)
                Y = y.reshape((N, 1))
                X = data[:, 0:-1]
            else:
                y = data[:, -1]
                N = len(y)
                y = y.reshape((N, 1))
                Y = np.column_stack((data[:,-dimY:-1], y))
                X = data[:, 0:-dimY]

        else :
            if dimY == 1:
                y = data[:, -1]
                N = len(y)
                Y = y.reshape((N, 1))
                Y,K = self.one_hot_encode(Y)
                X = data[:, 0:-1]
            else:
                K = [None]*dimY
                y = data[:, -1]
                N = len(y)
                Y = y.reshape((N, 1))
                Y,K[0] = self.one_hot_encode(Y)

                for i in (2,dimY+1):
                    y = data[:, -i]
                    y = y.reshape((N, 1))
                    y, K[i-1] = self.one_hot_encode(y)
                    Y = np.column_stack((y, Y))
                K = np.asarray(K)
                X = data[:, 0:-dimY]

        return X,Y,K

    def one_hot_encode(self,y):
        values = np.unique(y)
        return np.array([values == i for i in y]),len(values)

    def buildLayers(self, nHiddenLayers, nNeurons, D, K):
        W = []
        b = []
        if nHiddenLayers == 1:
            W.append(np.random.randn(D, nNeurons[0]))
            b.append(np.random.randn(nNeurons[0]))
            print("W input Shape ", W[0].shape)

            W.append(np.random.randn(W[-1].shape[1], K))
            b.append(np.random.randn(K))
            print("W output Shape ", W[-1].shape)
        else:

            W.append(np.random.randn(D,nNeurons[0]))
            b.append( np.random.randn(nNeurons[0]))
            print("W input Shape ", W[0].shape)

            for i in range(1,nHiddenLayers):
                W.append(np.random.randn(W[i-1].shape[1],nNeurons[i]))
                b.append(np.random.randn(nNeurons[i]))
                print("W layer ",i+1 , " Shape ", W[i].shape)

            W.append(np.random.randn(W[-1].shape[1], K))
            b.append(np.random.randn(K))
            print("W output Shape ", W[-1].shape)

        return W,b


    def feed_forward(self, X, W, b, act_fun,classification ):
        Z = []
        if classification == False:
            Z.append(self.activation(act_fun[0], (np.matmul(X,W[0]) +b[0])))
            for i in range(1, len(W)-1):
                Z.append(self.activation(act_fun[i],np.matmul(Z[i-1], W[i]) + b[i]))
            Y_hat =  np.matmul(Z[-1],W[-1])+b[-1]
        else:
            Z.append(self.activation(act_fun[0], (np.matmul(X, W[0]) + b[0])))
            for i in range(1, len(W) - 1):
                Z.append(self.activation(act_fun[i], np.matmul(Z[i - 1], W[i]) + b[i]))
            Y_hat = self.softmax(np.matmul(Z[-1], W[-1]) + b[-1])
        return Z, Y_hat

    def softmax(self,H):
        eH = np.exp(H)
        return eH / eH.sum(axis=1, keepdims=True)

    def shuffle_set(self,X,Y, bool = False):
        if bool == True:
            idx = np.random.permutation(len(Y))
            X = X[idx, :]
            Y = Y[idx, :]
        return X,Y


    def OLS(self, Y, Y_hat):
        return np.trace((Y - Y_hat).T.dot(Y - Y_hat))

    def cross_entropy(self,Y, P):
        return -np.sum(Y * np.log(P))

    def accuracy(self,Y, P):
        return np.mean(Y.argmax(axis=1) == P.argmax(axis=1))


    def R2(self, Y, Y_hat):
        return 1 - ((Y - Y_hat) ** 2).sum(axis=0) / ((Y - Y_hat.mean(axis=0)) ** 2).sum(axis=0)


    def activation(self, act_fun, H):
        #f defaults to relu
        f = H * (H > 0)
        if act_fun == "tanh":
            f = np.tanh(H)
        elif act_fun == "sigmoid":
            f = 1 / (1 + np.exp(-H))
        return f

    def d_act(self, act_fun, dZ,Z):
        # df defaults to d relu
        df = dZ*(Z>0)
        if act_fun == "tanh":
            df = dZ*(1 - Z*Z)
        elif act_fun == "sigmoid":
            df =dZ*(Z*(1 - Z))
        return df

    #----------------------------------------------------------------------------------------------------------------

    def fitData(self, X,Y,W,b, eta , batchS, mu, gamma, l1, l2, act_fun, epochs,classification, shuffle_data ):
        J = []
        vW = []
        vb = []
        Gb = []
        GW = []
        db = []

        for v in range (len(W)):
            vW.append(0)
            vb.append(0)
            GW.append(0)
            Gb.append(0)
            db.append(0)

        epsilon = 1e-13,

        if gamma ==1:
            epsilon = 1

        n_batches = len(Y) // batchS
        dh =[None]*len(W)
        dz=[None]*(len(W)-1)
        dw=[None]*len(W)

        if shuffle_data == True:
            X,Y = self.shuffle_set(X,Y, shuffle_data)

        X_train = X[:int(0.6 * X.shape[0]), :]
        Y_train = Y[:int(0.6 * Y.shape[0])]

        W = np.asarray(W)

        for epoch in range(1,int(epochs+1)):
            idx = np.random.permutation(len(Y_train))
            X_train = X_train[idx, :]
            Y_train = Y_train[idx, :]
            for i in range(n_batches):
                X_b = X_train[(i * batchS):((i + 1) * batchS), :]
                Y_b = Y_train[(i * batchS):((i + 1) * batchS), :]
                Z, Y_hat_b = self.feed_forward(X_b,W,b, act_fun,classification)
                if classification ==False:
                    J.append(self.OLS(Y_b, Y_hat_b) + (l2/2)*(sum(np.sum((dub*dub)) for dub in W))
                             + l1*(sum(np.sum(np.abs(dub))for dub in W)))
                else:
                    J.append(self.cross_entropy(Y_b, Y_hat_b) + (l2 / 2) * (sum(np.sum((dub * dub)) for dub in W))
                             + l1 * (sum(np.sum(np.abs(dub)) for dub in W)))

                dh[-1] = Y_hat_b - Y_b
                dw[-1] = np.matmul(Z[-1].T, dh[-1])+l1*np.sign(W[-1]) + l2 * W[-1]
                db[-1] = dh[-1].sum(axis=0)+l1*np.sign(b[-1]) + l2 * b[-1]
                GW[-1] = GW[-1] * gamma + (1 - gamma) * (dw[-1] ** 2)
                Gb[-1] = Gb[-1] * gamma + (1 - gamma) * (db[-1] ** 2)
                vW[-1] = mu * vW[-1] - (eta/ np.sqrt(GW[-1] + epsilon)) * dw[-1]
                vb[-1] = mu * vb[-1] - (eta/ np.sqrt(Gb[-1] + epsilon)) * db[-1].sum(axis=0)
                W[-1] += mu*vW[-1] - ((eta/ np.sqrt(GW[-1] + epsilon)) * (dw[-1]))
                b[-1] += mu*vb[-1] -((eta/ np.sqrt(Gb[-1] + epsilon))*(db[-1].sum(axis=0)))

                for dub in range(2, len(W)):
                    dz[-(dub-1)] = np.matmul(dh[-(dub-1)], W[-(dub-1)].T)
                    dh[-dub] =  self.d_act(act_fun[-(dub-1)],dz[-(dub-1)],Z[-(dub-1)])
                    dw[-dub] = np.matmul(Z[-(dub)].T, dh[-dub]) +l1*np.sign(W[-dub]) + l2*W[-dub]
                    db[-dub] = dh[-dub].sum(axis=0) +l1*np.sign(b[-dub]) + l2*b[-dub]
                    GW[-dub] = GW[-dub] * gamma + (1 - gamma) * (dw[-dub] ** 2)
                    Gb[-dub] = Gb[-dub] * gamma + (1 - gamma) * (db[-dub] ** 2)
                    vW[-dub] = mu * vW[-dub] - (eta/ np.sqrt(GW[-dub] + epsilon)) * dw[-dub]
                    vb[-dub] = mu * vb[-dub] - (eta/ np.sqrt(Gb[-dub] + epsilon)) * db[-dub].sum(axis=0)
                    W[-dub] += mu*vW[-dub] - ((eta/ np.sqrt(GW[-dub] + epsilon))*(dw[-dub]))
                    b[-dub] += mu*vb[-dub] -((eta/ np.sqrt(Gb[-dub] + epsilon))*(db[-dub].sum(axis = 0)))

                dz[0] = np.matmul(dh[1], W[1].T)
                dh[0] = self.d_act(act_fun[0],dz[0],Z[0])
                dw[0] = np.matmul(X_b.T, dh[0])+l1*np.sign(W[0]) + l2*W[0]
                db[0] = dh[0].sum(axis=0)+l1*np.sign(b[0]) + l2*b[0]
                GW[0] = GW[0] * gamma + (1 - gamma) * (dw[0] ** 2)
                Gb[0] = Gb[0] * gamma + (1 - gamma) * (db[0] ** 2)
                vW[0] = mu * vW[0] - (eta/ np.sqrt(GW[0] + epsilon)) * dw[0]
                vb[0] = mu * vb[0] - (eta/ np.sqrt(Gb[0] + epsilon)) * db[0].sum(axis=0)
                W[0] += mu*vW[0] -((eta/ np.sqrt(GW[0] + epsilon)) * (dw[0]))
                b[0] += mu*vb[0] -((eta/ np.sqrt(Gb[0] + epsilon)) * (db[0].sum(axis=0)))

            if epoch% (int((epochs*.05)*100)/100) == 0:
                Y_hat_train = self.feed_forward(X_train, W, b,act_fun,classification)[-1]
                if classification == False:
                    print("Epochs at %",int(100*epoch/epochs), " | R squared train = %",
                          np.round(100000*(self.R2(Y_train,Y_hat_train)[0]))/1000)
                else:
                    print("Epochs at %", int(100 * epoch / epochs), " | Accuracy train = %",
                          np.round(100000 * (self.accuracy(Y_train, Y_hat_train))) / 1000)

        X_cv = X[int(0.6 * X.shape[0]):int(0.8 * X.shape[0]),:]
        Y_cv = Y[int(0.6 * Y.shape[0]):int(0.8 * Y.shape[0])]

        X_test = X[int(0.8 * X.shape[0]):,:]
        Y_test = Y[int(0.8 * Y.shape[0]):]

        Y_hat_train = self.feed_forward(X_train,W,b,act_fun,classification)[-1]
        Y_hat_cv = self.feed_forward(X_cv, W, b,act_fun,classification)[-1]
        Y_hat_test = self.feed_forward(X_test, W, b,act_fun,classification)[-1]

        if classification == False:
            plt.plot(J[::100])
            plt.title("R squared train = %{}".format(np.round(100000*(self.R2(Y_train,Y_hat_train)[0]))/1000))
            plt.show()
            print( "R squared train = %",100*self.R2(Y_train,Y_hat_train)[0],
                   ", R squared val = %", 100*self.R2(Y_cv,Y_hat_cv)[0],", R squared test = %",100*self.R2(Y_test,Y_hat_test)[0])
        else:
            plt.plot(J[::100])
            plt.title("Accuracy train = %{}".format(np.round(100000 * (self.accuracy(Y_train, Y_hat_train))) / 1000))
            plt.show()
            print("Accuracy train = %", 100 * self.accuracy(Y_train, Y_hat_train),
                  ", Accuracy val = %", 100 * self.accuracy(Y_cv, Y_hat_cv), ", Accuracy test = %",
                  100 * self.accuracy(Y_test, Y_hat_test))

