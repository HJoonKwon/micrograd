from micrograd.engine import Value
import random


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, act="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.act = act

    def __call__(self, x):
        assert len(x) == len(self.w)
        acts = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.act == "tanh":
            acts = acts.tanh()
        elif self.act == "relu":
            acts = acts.relu()
        return acts

    def __repr__(self):
        return f"{self.act if self.act is not None else 'Linear'}Neuron({len(self.w)})"

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nouts, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nouts)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, act="tanh"):
        nouts = nouts if isinstance(nouts, list) else list(nouts)
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], act=None if i == len(nouts) - 1 else act)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
