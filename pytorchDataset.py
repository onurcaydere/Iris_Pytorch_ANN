import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Verinin Yüklenmesi
data=pd.read_csv("C:/Users/Onur Çaydere/Desktop/Pytorch/PytorchUdemyBootcamp/data/iris.csv")
data=data.drop("Id",axis=1)
# Features Input and Features Output
input_features=data.iloc[:,data.columns!="Species"].values

output_features=data.Species.values

one=LabelEncoder() # Str ifadeyi işlemlerim kolay olsun diye label encoder ile 0 1 2 değerleine yerleştirdim.
output_features=one.fit_transform(output_features)

# Train Test Split
x_train,x_test,y_train,y_test=train_test_split(input_features,output_features,test_size=0.1,random_state=42)

# Float32 for numpy array
x_train,x_test=x_train.astype(np.float32),x_test.astype(np.float32)
y_train,y_test=y_train.astype(np.float32),y_test.astype(np.float32)
print(x_train.shape)

# Tensor from numpy

x_train_T=torch.from_numpy(x_train)
x_test_T=torch.from_numpy(x_test)
y_train_T=torch.from_numpy(y_train).type(torch.LongTensor)
y_test_T=torch.from_numpy(y_test).type(torch.LongTensor)

# TensorDataSet
train=TensorDataset(x_train_T,y_train_T)
test=TensorDataset(x_test_T,y_test_T)

# Bacth_Size ,Epoch
batch_size=125
epochs=200

train_loader=DataLoader(train,batch_size,shuffle=True)
test_loader=DataLoader(test,batch_size,shuffle=True)


#Create ANN Model

class Model(nn.Module):
    def __init__(self,input_features=4,h1=64,h2=128,out_features=3):
        super(Model,self).__init__()
        self.fc1=nn.Linear(input_features, h1)
        self.fc2=nn.Linear(h1, h2)
        self.fc4=nn.Linear(h2, out_features)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))

        x=self.fc4(x)
        
        return x
    
    
# Create Optimizer and  Loss Function
model=Model()

error=nn.CrossEntropyLoss()

optimizer=torch.optim.SGD(model.parameters(),lr=0.002)

losses=[]

for epoch in range(epochs):
    for i,(data,label) in enumerate(train_loader):
     
        optimizer.zero_grad()


        output=model(data)
        loss=error(output,label)

        loss.backward()

        optimizer.step()
        
        losses.append(loss.data)
    if epoch%10==0:
        print("Epoch {} Loss {} ".format(epoch,loss.data))

correct=0
total=0

with torch.no_grad():
    for i ,(data,labels) in enumerate(test_loader):
        y_pred=model(data)
        prediction=torch.max(y_pred,1)[1]
        correct+=(prediction==labels).sum()
        total+=len(labels)
        test_loss=error(y_pred,labels)
    acc=100*correct/float(total)
print(test_loss)
print(acc)

# Model Save

torch.save(model.state_dict(),"C:/Users/Onur Çaydere/Desktop/Pytorch/PytorchUdemyBootcamp/data/iris_model.pt")


# Model Callbacks

new_model=Model()
new_model.load_state_dict(torch.load("C:/Users/Onur Çaydere/Desktop/Pytorch/PytorchUdemyBootcamp/data/iris_model.pt"))
new_model.eval()

