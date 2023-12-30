import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")
        return out
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Not supported type. Only int/float are supported for now"
        out = Value(self.data ** other, _children=(self, other), _op=f"**{other}")
        return out 
    
    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other 

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other 
    
    def exp(self):
        out = Value(math.exp(self.data), _children=(self,), _op="exp")
        return out

    def tanh(self):
        exp = self.exp()
        negexp = (-self).exp() 
        return (exp - negexp) / (exp + negexp)

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, _children=(self,), _op="ReLU")
        return out  

    def backward(self):
        pass
