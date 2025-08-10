from network import MLP,Layer,Neuron
from ML import Value
import numpy as np
import idx2numpy
def fit_list(list):
    fitted=[]
    for i in list:
        for y in i:
            fitted.append(y)
    return fitted
file = "data/train.idx3-ubyte"
file2= "data/labels.idx1-ubyte"
print("PROGRAM START.")
temp_xs = idx2numpy.convert_from_file(file)
temp_ys = idx2numpy.convert_from_file(file2)
ys=[]
xs=[]
print("Getting the labels and features ready..")
for i in range(100):
    ys.append(temp_ys[i].item())
    xs.append(temp_xs[i].tolist())
for i in range(len(xs)):
    xs[i]=fit_list(xs[i])
#TMP here stands for Trained Model Parameters
TMP = []
print("Modifying parameters.")
with open("trained_model_parameters.txt","r") as f:
    for line in f:
        TMP.append((float(line[:-2])))
print("Parameters added.")
#TM stands for Trained Model
NN=MLP(784,[32,1])
NN.changeparam(TMP)
# print(TM.parameters()[0])
print("Testing started")
for i in range(100):
    print(f'pred:{NN(xs[i])} label: {ys[i]} Valid: {NN(xs[i])[0].data == ys[i]}')
