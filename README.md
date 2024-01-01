# micrograd

## motivations 
Self-study notes for Andrej Karpathy's micrograd lecture
- Youtube lecture: [link](https://youtu.be/VMj-3S1tku0?si=hRNZvFtXWlAlIXAF)
- Source code: [link](https://github.com/karpathy/micrograd)

## notes 
Even though the concepts here are pretty simple and most ML engineers are probably already familiar with them, I had a blast going over everything from computational graphs to backprop and building simple neural networks from scratch. I tweaked a small part of the original source code just for fun, to play around with the code base. The unit tests are different from the original ones too, because I wanted to create them myself, just to deepen my understanding.

## tests 
```bash 
python -m pytest 
```

## What I learned

1) **Multivariate derivatives**. While differentiating multivariate equations with mathematical expressions is straightforward, it can be prone to errors when coding. Similar to how Andrej did not edit out his mistakes during the lecture, using them instead as teaching moments, I concur that these are common pitfalls. Below, I've included a snippet from ```engine.py``` to illustrate this point. In a step of gradient descent, grad should accumulate to encompass all gradients in a multivariate context. For more information, refer to [this article](https://en.wikipedia.org/wiki/Chain_rule#Multivariable_case) on the chain rule for multivariate functions.
   
```python
def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad ## <--------- See '+=' 
            other.grad += self.data * out.grad ## <--------- See '+='

        out._backward = _backward
        return out
```

2) **Topological sort for gradients**. Every operation performed on incoming data, leading up to the loss, can be depicted as a computational graph (i.e., a Directed Acyclic Graph). This allows us to leverage graph theories to simplify our tasks. In micrograd, ```Value``` represents a node in this graph and tracks its child nodes, which constitute the current node. Therefore, if we perform a topological sort on the nodes in this graph, the order of this sort aligns with the sequence for computing gradients. For instance, we start calculating gradients from the loss node, whose gradient is initialized as ```1.0```, because it is the prerequisite to calculate its other nodes' gradients.

3) **Saving a backward function as a class attribute**. Once we define operations for the ```Value``` class, we define a ```_backward``` function for each operation, which is stored as an attribute of the output ```Value``` resulting from that operation. Subsequently, when we invoke ```backward()``` at the loss node to calculate the gradients of all trainable parameters, the ```backward()``` method calls the stored ```_backward``` functions. I find this approach very straightforward and easy to understand. Good to learn about this process. 
