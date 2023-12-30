import torch 
from micrograd.nn import Neuron, Layer, MLP

def test_forward():
    nin = 3 
    nouts = [4, 4, 3, 1]
    mlp = MLP(nin, nouts)
    xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
    ] 
    ys = [] 
    for x in xs:
        y = mlp(x)
        ys.append(y)
    
    