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
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "Not supported type. Only int/float are supported for now"
        out = Value(self.data**other, _children=(self,), _op=f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
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

        def _backward():
            self.grad += math.exp(self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        exp = self.exp()
        negexp = (-self).exp()
        return (exp - negexp) / (exp + negexp)

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, _children=(self,), _op="ReLU")

        def _backward():
            self.grad += 0 if self.data < 0 else out.grad

        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0
        visit = set()
        topo = []

        def topo_sort(n):
            assert n != -1, f"weird n = {n}, topo={topo}"
            if n in visit:
                return
            visit.add(n)
            for child in n._prev:
                topo_sort(child)
            topo.append(n)

        topo_sort(self)
        topo_rev = reversed(topo)
        for n in topo_rev:
            n._backward()
