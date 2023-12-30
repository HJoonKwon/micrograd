import torch
import numpy as np
from micrograd.engine import Value

def test_ops():
    a = Value(-4.0)
    b = Value(8.0)

    a_ = torch.tensor([-4.0])
    b_ = torch.tensor([8.0])

    assert np.allclose((a + b).data, (a_ + b_).item())
    assert np.allclose((a * b).data, (a_ * b_).item())
    assert np.allclose((a**b.data).data, (a_**b_).item())
    assert np.allclose(
        (a.tanh() * b.exp()).data, (torch.tanh(a_) * torch.exp(b_)).item()
    )
    assert np.allclose(
        ((5 + a) + (5 - b) * (6 * a)).data, ((5 + a_) + (5 - b_) * (6 * a_)).item()
    )
    assert np.allclose(a.relu().data, (a_.relu()).item())

def test_backward():
    a = Value(-4.0)
    b = Value(8.0)
    c = Value(12.0)
    e = Value(4.0)
    d = (a * b) / c / e + (b / 12.0).relu() * e ** 2 / 36.0 
    f = d.tanh() + a.relu()    
    f.backward()

    a_ = torch.tensor([-4.0], requires_grad=True)
    b_ = torch.tensor([8.0], requires_grad=True)
    c_ = torch.tensor([12.0], requires_grad=True)
    e_ = torch.tensor([4.0], requires_grad=True)
    d_ = (a_ * b_) / c_ / e_ + (b_ / 12.0).relu() * e_ ** 2 / 36.0 
    f_ = d_.tanh() + a_.relu() 
    f_.backward() 
    
    np.allclose(a.grad, a_.grad.item())
    np.allclose(b.grad, b_.grad.item())
    np.allclose(c.grad, c_.grad.item())
    np.allclose(e.grad, e_.grad.item())
    
