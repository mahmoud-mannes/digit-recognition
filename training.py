from network import MLP,Neuron,Layer
from ML import Value
from random import randint
#Unfortunately had to use numpy for the training portion as the MNIST dataset is in a weird file format.
import numpy as np
import idx2numpy
def fit_list(list):
    fitted=[]
    for i in list:
        for y in i:
            fitted.append(y)
    return fitted
def random_batch(image,label,size_batch):
    image_batch=[]
    label_batch=[]
    for i in range(size_batch):
        index=randint(0,len(image))
        image_batch.append(image[index].tolist())
        label_batch.append(label[index].item())
    return image_batch,label_batch
file = "data/train.idx3-ubyte"
file2= "data/labels.idx1-ubyte"
full_xs = idx2numpy.convert_from_file(file)
full_ys= idx2numpy.convert_from_file(file2)
full_batch=random_batch(full_xs,full_ys,50)
xs= (full_batch[0])
for i in range(len(xs)):
    xs[i]=fit_list(xs[i])
ys= (full_batch[1])
NN=MLP(784,[28,28,1])
print("Training start.")
NN.train(xs,ys)
trained_model_parameters=NN.parameters()
with open('trained_model_parameters.txt','w') as f:
    for i in trained_model_parameters:
        f.write(str(i.data)+"\n")