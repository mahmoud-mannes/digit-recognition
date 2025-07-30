import random
from ML import Value
class Neuron:
    def __init__(self,nin,nonlin=True):
        self.w=[Value(random.uniform(-1,1)) for i in range(nin)]
        self.b = Value(0)
        self.nonlin=nonlin
    def __call__(self,x):
        activ=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        
        activ = activ.ReLU() if self.nonlin else activ
        return activ
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    def parameters(self):
        return self.w+[self.b]
class Layer:
    def __init__(self,nin,nout,nonlin=True):
        self.neurons=[Neuron(nin,nonlin) for i in range(nout)]
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out
    def __repr__(self):
        return f"Layer of: {self.neurons}"
    def parameters(self):
        l=[]
        for i in self.neurons:
            for p in i.parameters():
                l.append(p)
        return l
class MLP:
    def __init__(self,nin,nouts):
        sz = [nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return [x[i] for i in range(len(x))]
    def __repr__(self):
        
        output="\n".join([str(self.layers[i]) for i in range(len(self.layers))])
        return "MLP with layers: \n" + output
    def parameters(self):
        return [p for i in self.layers for p in i.parameters()]

    def train(self,xs,ys,final_epoch=1,step=0.001):
        epoch=0
        ypred=[self(xs[i]) for i in range(len(xs))]
        loss=(sum((ypred[i][0]-ys[i])**2 for i in range(len(ys))))
        while epoch <final_epoch:
            v=loss.backward()
            for i in self.parameters():
                i.data+=-step*i.grad
                i.grad=0
            epoch+=1
            print(f'Epoch number: {epoch}')
        ypred=[self(xs[i]) for i in range(len(xs))]
        loss=(sum((ypred[i][0]-ys[i])**2 for i in range(len(ys))))
        print(loss)
    def changeparam(self,newparams):
        ind=0
        for i in self.parameters():
            i.data=newparams[ind]
            ind+=1
        
