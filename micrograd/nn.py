from micrograd.engine import Value
import random


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        assert len(x) == len(self.w)
        acts = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return acts.relu() if self.nonlin else acts

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nouts, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nouts)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        nouts = nouts if isinstance(nouts, list) else list(nouts)
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
