import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    classes = np.unique(y)
    means = np.zeros([classes.shape[0],X.shape[1]])

    for i in range(classes.shape[0]):
        means[i,] = np.mean(X[np.where(y == classes[i])[0],],axis=0)

    covmat = np.cov(X.transpose())

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    covmats = []
    classes = np.unique(y)
    means = np.zeros([classes.shape[0],X.shape[1]])

    for i in range(classes.shape[0]):
        means[i,] =np.mean(X[np.where(y==classes[i])[0],],axis=0)
        covmats.append(np.cov(np.transpose(X[np.where(y==classes[i])[0],])))

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    res = np.zeros((Xtest.shape[0],means.shape[0]))
    f = 1/np.sqrt((2*pi)**means.shape[1]*det(covmat))
    for j in range(means.shape[0]):
        res[:,j] = f * np.exp(-0.5*np.array([np.dot(np.dot((Xtest[i,:] - means[j,:]),inv(covmat)),np.transpose(Xtest[i,:] - means[j,:])) for i in range(Xtest.shape[0])]))

    ypred = np.argmax(res,axis=1) + 1
    res = (ypred == ytest.ravel())
    acc = len(np.where(res)[0])
    acc = float(acc)/len(ytest)

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    res = np.zeros((Xtest.shape[0],means.shape[0]))
    for j in range(means.shape[0]):
        f = 1/np.sqrt((2*pi)**means.shape[1]*det(covmats[j]))
        res[:,j] = f * np.exp(-0.5*np.array([np.dot(np.dot((Xtest[i,:] - means[j,:]),inv(covmats[j])),np.transpose(Xtest[i,:] - means[j,:])) for i in range(Xtest.shape[0])]))
    ypred = np.argmax(res,axis=1) + 1
    res = (ypred == ytest.ravel())
    acc = len(np.where(res)[0])
    acc = float(acc)/len(ytest)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    x_transpose = np.transpose(X)
    x_transpose_x = np.dot(x_transpose , X)
    x_transpose_y = np.dot(x_transpose , y)
    inverse = np.linalg.inv(x_transpose_x)
    w = np.dot(inverse, x_transpose_y)

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD

    N = Xtest.shape[0]

    #transpose(w)*X predicted by model
    predicted_output = np.dot(Xtest,w)

    #Difference of the predicted output and the actual output
    error = np.subtract(ytest, predicted_output)

    #Taking the square of the error
    error_squared = np.multiply(error, error)

    #Mean sqaured error normalised
    mse = (np.sum(error_squared))/N

    return mse

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    N, d = np.shape(X)
    dot_XT_X = np.dot(X.T,X)
    prod_lambda_I = lambd*(np.eye(d))
    sum_XT_X_lambda_I = dot_XT_X + prod_lambda_I
    inv_sum_XT_X_lambda_I = np.linalg.inv(sum_XT_X_lambda_I)

    dot_XT_y = np.dot(X.T,y)
    w = np.dot(inv_sum_XT_X_lambda_I,dot_XT_y)

    return w

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    N = X.shape[0]

    error = 0

    error = np.subtract(y.T, np.dot(w.T, X.T))

    error_squared = np.multiply(error, error)


    error_sum = np.sum(error_squared)


    reg_term = np.multiply(lambd, np.dot(w.T, w))

    error = (error_sum / 2) + (reg_term / 2)



    lambda_w = np.multiply(lambd, w)

    transposey_X = np.dot(y.T, X)

    transposeX_X = np.dot(X.T, X)

    product_W_X =  np.dot(w.T, transposeX_X)


    difference = np.subtract(lambda_w, transposey_X)

    error_grad = np.add(difference, product_W_X)


    error = error.flatten()
    error_grad = error_grad.flatten()

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))

    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    Xp = np.ones((N, p + 1));
    for i in range(1, p + 1):
        Xp[:, i] = x ** i;

    return Xp

# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')


# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle_train = testOLERegression(w,X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
weights = w_i
mle_test = testOLERegression(w_i,X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept on train data '+str(mle_train))
print('MSE with intercept on train data '+str(mle_test))
print('MSE without intercept on test data '+str(mle))
print('MSE with intercept on test data '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
optimal_mse = 5000
optimal_w = None
l = 0
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if(mses3[i] < optimal_mse):
        optimal_mse = mses3[i]
        optimal_mset = mses3_train[i]
        optimal_w = w_l
        l = lambd
    i = i + 1

print("MSE for optimal value of lambda on train data " + str(optimal_mset))
print("MSE for optimal value of lambda on test data " + str(optimal_mse))
print("Optimal lambda{}".format(l))
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.show()

ode_weights = weights.flatten()
ridge_Weights = optimal_w.flatten()
print(np.linalg.norm(ode_weights-ridge_Weights, ord=None, axis=None, keepdims=False))


plt.plot(range(len(weights.flatten())),weights.flatten())
plt.title('Weights using OLE')
plt.show()
plt.plot(range(len(optimal_w.flatten())),optimal_w.flatten())
plt.title('Weights using Ridge regression')
#plt.legend(['OLE','Ridge regression'])
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    if(lambd == 0.06):
        print("MSE for optimal value of lambda on train data " + str(mses4_train[i]))
        print("MSE for optimal value of lambda on test data " + str(mses4[i]))
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
optimal_mse5 = float('inf')
optimal_mse5_1 = float('inf')
optimal_p = None
optimal_p_0 = None
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    if mses5[p,0] < optimal_mse5_1:
        optimal_mse5_1 = mses5[p,0]
        optimal_p_0 = p
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    if mses5[p,1] < optimal_mse5:
        optimal_mse5 = mses5[p,1]
        optimal_p = p

print("MSE train for p for lambda 0 "+str(p)+":"+str(mses5_train[p,0]))
print("MSE test for p for lambda 0 "+str(p)+":"+str(mses5[p,0]))
print("MSE train for p for lambda 0.06 "+str(p)+":"+str(mses5_train[p,1]))
print("MSE test for p for lambda 0.06 "+str(p)+":"+str(mses5[p,1]))
print("optimal p for lambda 0 " +str(optimal_p_0))
print("optimal p for lambda 0.06 "+str(optimal_p))
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
