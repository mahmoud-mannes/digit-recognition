import math
def softmax(self):
    max_input=max([i.data for i in self])
    exps= [math.exp(z.data-max_input) for z in self]
    sum_exps=sum(exps)
    out= [Value(exps[e]/sum_exps,{self[e]},'softmax') for e in range(len(exps))]
    for i in out:
        i._backward=softmax_grad
    return out
def softmax_grad(self,target,out):
    self.grad+=(self-target)*out.grad
class Value:
    def __init__(self,data,_children=set(),op_=""):
        self.data=data
        self._backward = lambda a,b,c: None
        self._children=_children
        self.op_=op_
        self.grad=0.0
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        res=self.data+other.data
        out=Value(res,(self,other),'+')
        
        def _backprop(a,b,c):
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backprop
        return out
    def __radd__(self,other):
        return self+other
    def __repr__(self):
        return "Value: ("+str(self.data)+")"
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        res=self.data*other.data
        out=Value(res,(self,other),"*")
        def _backprop(a,b,c):
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backprop
        return out
    def __rmul__(self,other):
        return self*other
    def __gt__(self,other):
        if isinstance(self,int) or isinstance(self,float):
            self=Value(self)
        if isinstance(other,float) or isinstance(other,int):
            other=Value(other)
        return self.data>other.data
    def __neg__(self):
        res=-self.data
        return Value(res)
    def __sub__(self,other):
        return self + (-other)
    def __pow__(self,other):
        out=Value(self.data**other,(self,),"**")
        def _backprop(a,b,c):
            #This should be self.data ** (other -1), however I've replaced it with self.data
            #because the only power I used was 2, so (other -1)= (2-1)=1, so there's no need
            #for adding a **(other-1) as it'll only make the calculations slower.
            self.grad+=other*(self.data)*out.grad
        out._backward=_backprop
        return out
    def __lt__(self,other):
        #Ran into errors where an int is compared to a Value class. (ints don't have a data attribute)
        if isinstance(self,int) or isinstance(self,float):
            self=Value(self)
        if isinstance(other,float) or isinstance(other,int):
            other=Value(other)
        return self.data<other.data
    def tanh(self):
        if isinstance(self,int) or isinstance(self,float):
            x = max(min(Value(self), Value(50)), Value(-50))  # Clip to prevent overflow
        else:
            x = max(min(self, Value(50)), Value(-50))
        #Ran into issues where the value of x was contained within multiple Value classes e.g Value(Value(2.0))
        while isinstance(x,Value):
            x=x.data
        e_pos = math.exp(x)
        e_neg = math.exp(-x)
        res=(e_pos - e_neg) / (e_pos + e_neg)
        out=Value(res,{self},'tanh')
        def _backprop(a,b,c):
            self.grad+=(1-res**2)*out.grad
        out._backward=_backprop
        return out
    def ReLU(self):
        return Value(0.0) if self.data < 0 else self
    def backward(self,pred,target):
        v=[]
        #Creating a list that includes all of the children of the current neuron we're backpropagating through.
        def makelist(x):
            visited=set()
            topo=[]
            def build(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._children:
                        build(child)
                    topo.append(v)
            build(x)
            return topo
        v=makelist(self)
        self.grad=1.0
        for i in reversed(v):
            i._backward(pred,target,i)
        return v
