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
