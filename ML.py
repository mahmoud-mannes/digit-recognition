from math import exp
class Value:
    def __init__(self,data,_children=set(),op_=""):
        self.data=data
        self._backward = lambda: None
        self._children=_children
        self.op_=op_
        self.grad=0.0
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        res=self.data+other.data
        out=Value(res,(self,other),'+')
        
        def _backprop():
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
        def _backprop():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backprop
        return out
    def __rmul__(self,other):
        return self*other
    def __neg__(self):
        res=-self.data
        return Value(res)
    def __sub__(self,other):
        return self + (-other)
    def __pow__(self,other):
        out=Value(self.data**other,(self,),"**")
        def backprop():
            self.grad+=other*self.data**(other-1)*out.grad
        out._backward=backprop
        return out
    def tanh(self):
        res=(exp(2*self.data)-1)/(exp(2*self.data)+1)
        out=Value(res,{self},'tanh')
        def backprop():
            self.grad+=(1-res**2)*out.grad
        out._backward=backprop
        return out
    def ReLU(self):
        return Value(0.0) if self.data < 0 else self
#     def makelist(self):
#         v=[]
#         if self not in v:
#             v.append(self)
#             for i in self._children:
#                 i.makelist()
#         return v
    def backward(self):
#         v=self.makelist()
        v=[]
        def makelist(x):
            v=[]
            if x not in v:
                v.append(x)
                for i in x._children:
                    v.extend(makelist(i))
            return v
        v=makelist(self)
        self.grad=1.0
        for i in v:
            i._backward()
        return v