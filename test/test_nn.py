from micrograd.nn import Neuron, Layer, MLP


def test_forward():
    nin = 3
    nouts = [4, 4, 3, 1]
    mlp = MLP(nin, nouts)
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [mlp(x) for x in xs]


def test_backward():
    nin = 3
    nouts = [4, 4, 3, 1]
    mlp = MLP(nin, nouts, "tanh")
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    prev_loss = float("inf")
    for _ in range(10):
        ypreds = [mlp(x) for x in xs]
        loss = sum((y - y_pred) ** 2 for y, y_pred in zip(ys, ypreds)) / len(ypreds)
        assert prev_loss > loss.data
        prev_loss = loss.data
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.01 * p.grad
