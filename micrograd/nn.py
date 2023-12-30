from micrograd.engine import Value
import random


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self):
        pass

    def __call__(self):
        pass

    def parameters(self):
        pass


class Layer(Module):
    def __init__(self):
        pass

    def __call__(self):
        pass

    def parameters(self):
        pass


class MLP(Module):
    def __init__(self):
        pass

    def __call__(self):
        pass

    def parameters(self):
        pass
