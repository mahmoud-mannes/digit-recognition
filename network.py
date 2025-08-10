import random
from ML import Value,softmax
class Neuron:
    def __init__(self,nin,final=False):
        self.w = [Value(random.uniform(-1,1)) for i in range(nin)]
        self.b = Value(0)
        self.final = final
    def __call__(self,x):
        activ = self.b
        #Weighted sum to calculate the activation of the neuron.
        for wi, xi in zip(self.w, x):
            activ = activ+(wi * xi)
        self.activ = activ.tanh()
        return self.activ
    def __repr__(self):
        return f"{'softmax' if self.final else 'tanh'}Neuron({len(self.w)})"
    def parameters(self):
        return self.w+[self.b]
class Layer:
    def __init__(self,nin,nout,final=False):
        self.final=final
        self.neurons=[Neuron(nin,final) for i in range(nout)]
    def __call__(self,x):
        #If current layer is an output layer, use softmax else use tanh
        if self.final:
            out = [n(x) for n in self.neurons]
            out2 = softmax([neuron.activ for neuron in self.neurons])
            for i in range(len(self.neurons)):
                self.neurons[i].activ=(out2[i])
            return out2
        else:
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
        self.layers=[Layer(sz[i],sz[i+1], final=(i==len(nouts)-1)) for i in range(len(nouts))]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return [x[i] for i in range(len(x))]
    def __repr__(self):
        
        output="\n".join([str(self.layers[i]) for i in range(len(self.layers))])
        return "MLP with layers: \n" + output
    def parameters(self):
        return [p for i in self.layers for p in i.parameters()]

    def train(self,xs,ys,step=0.005):
        print("Forward prop start.")
        y_pred=[self(xs[i]) for i in range(len(xs))]
        print("Predictions complete, loss calculation initiated.")
        loss = sum((y_pred[i][j]-ys[i][j])**2 for j in range(10) for i in range(len(ys))) * (1/len(ys))
        print(f'Loss before training is equal to {loss}')
        for i in range(len(y_pred)):
            print("Starting backward propagation.")
            for j in range(10):
                loss=((y_pred[i][j]-ys[i][j])**2)
                loss.backward(list(y_pred[i][j]._children)[0],ys[i][j])
                print([(children,children.grad,children.op_,children._backward) for children in loss._children[0]._children])
                print([(children,children.grad,children.op_,children._backward) for children in loss._children[0]._children[0]._children])
                
                for l in self.parameters():
                    l.data+=step*l.grad
                    l.grad=0
        y_pred=[self(xs[i]) for i in range(len(xs))]
        print("Predictions complete, loss calculation initiated.")
        loss = sum((y_pred[i][j]-ys[i][j])**2 for j in range(10) for i in range(len(ys))) * (1/len(ys))
        print(f'Loss after training is equal to {loss}')
    def changeparam(self,newparams):
        ind=0
        for i in self.parameters():
            i.data=newparams[ind]
            ind+=1

