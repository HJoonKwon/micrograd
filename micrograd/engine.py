class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data 
        self._prev = set(_children)
        self._op = _op 
        self.label = label 
        self._backward = lambda: None 
        self.grad = 0.0 
        
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self):
        pass 
    
    def __mul__(self):
        pass 
    
    def __sub__(self):
        pass 
    
    def __truediv__(self):
        pass 
    
    def exp(self):
        pass 
    
    def __neg__(self):
        pass 
    
    def __pow__(self):
        pass 
    
    def __radd__(self):
        pass 
    
    def __rsub__(self):
        pass 
    
    def __rmul__(self):
        pass 
    
    def tanh(self):
        pass 
    
    def relu(self):
        pass
    
    def backward(self):
        pass 