import numpy as py
import pandas as pd
import math
import matplotlib.pyplot as plt

idx = ['x1','x2','x3','x4','types']
df = pd.read_csv('iris.csv',names=idx)

data = df.head(100).values.tolist()

for i in data:
	if(i[4]=='Iris-setosa'):
		i.append(0)
	else:
		i.append(1)

arrA = data[0:10] + data[50:60]
arrB = data[10:20] + data[60:70]
arrC = data[20:30] + data[70:80]
arrD = data[30:40] + data[80:90]
arrE = data[40:50] + data[90:100]

train1 = arrA[:] + arrB[:] + arrC[:] + arrD[:]
train2 = arrB[:] + arrC[:] + arrD[:] + arrE[:]
train3 = arrA[:] + arrC[:] + arrD[:] + arrE[:]
train4 = arrA[:] + arrB[:] + arrD[:] + arrE[:]
train5 = arrA[:] + arrB[:] + arrC[:] + arrE[:]

val1 = arrE[:]
val2 = arrA[:]
val3 = arrB[:]
val4 = arrC[:]
val5 = arrD[:]

traindata = [train1, train2, train3, train4, train5]
valdata = [val1, val2, val3, val4, val5]

theta = [0.5,0.5,0.5,0.5]
list_theta = [theta[:] for i in range(5)]
dtheta = [0.5,0.5,0.5,0.5]
bias = 0.5
list_bias = [bias for i in range(5)]
dbias = 0

predict_train = []
predict_val = []
errors_train = []
errors_val = []

error_train_final = []
error_val_final = []

acc_train_final = []
acc_val_final = []

def Result(x,j):
  res = 0
  for i in range(4):
    global list_theta
    res += (x[i]*list_theta[j][i])
    
  global bias
  return res + bias

def Sigmoid(res):
  return 1/(1+math.exp(-res))

def Predict(act):
  if(act>0.5):
    return 1
  else:
    return 0

def Error(type,act):
  return math.pow((type-act),2)

def DthetaUpdate(x,trg,act):
  global dtheta
  for i in range(4):
    dtheta[i] = 2 * x[i] * (trg-act) * (1-act) * act

def DbiasUpdate(trg,act):
  global dbias
  dbias = 2 * (trg-act) * (1-act) * act

def ThetaUpdate(lr,j):
  global list_theta
  for i in range(4):
    list_theta[j][i] = list_theta[j][i] + (lr*dtheta[i])

def BiasUpdate(lr,j):
  global list_bias
  list_bias[j] += (lr*dbias)

for i in range(300):
  sum_err_train = 0
  sum_err_val = 0 
  sum_acc_train = 0
  sum_acc_val = 0 
  
  for j in range(5):
    sun = 0
    sun2 = 0
    tptn = 0
    tptn2 = 0
    for k in range(80):
    #   train
      act = Sigmoid(Result(traindata[j][k],j))

    #   predict_train.append(Predict(act))
      pred = Predict(act)

      if(pred==traindata[j][k][5]):
        tptn+=1

      
      sun += Error(traindata[j][k][5],act)

      DthetaUpdate(traindata[j][k][0:4],traindata[j][k][5],act)
      DbiasUpdate(traindata[j][k][5],act)

      ThetaUpdate(0.1,j)
      BiasUpdate(0.1,j)
 

    sum_err_train += sun/80 
    sum_acc_train += (tptn/80)*100
    
  
    for k in range(20):
    #   validation
      act = Sigmoid(Result(valdata[j][k],j))
      pred = Predict(act)

      if(pred==valdata[j][k][5]):
        tptn2+=1
      
      sun2 += Error(valdata[j][k][5],act)
    sum_err_val += sun2/20
    sum_acc_val += (tptn2/20)*100
  
  error_train_final.append(sum_err_train/5)
  error_val_final.append(sum_err_val/5)
  acc_train_final.append(sum_acc_train/5)
  acc_val_final.append(sum_acc_val/5)

plt.figure(1)
plt.plot(acc_train_final,'r-', label='train')
plt.plot(acc_val_final,'y-', label='validasi')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='upper right')

plt.figure(2)
plt.plot(error_train_final,'r-', label='training')
plt.plot(error_val_final,'y-', label='validasi')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='upper left')
plt.show()