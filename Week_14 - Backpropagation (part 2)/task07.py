from __future__ import annotations
import graphviz

import numpy as np


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label='') -> None:
        self.grad = 0.0
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f'Value(data={self.data})'

    def __add__(self, rhs: Value) -> Value:
        return Value(self.data + rhs.data, _children=(self, rhs), _op='+')

    def __mul__(self, rhs: Value) -> Value:
        return Value(self.data * rhs.data, _children=(self, rhs), _op='*')

    def tanh(self) -> Value:
        data = (np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)
        return Value(data=data, _children=(self, ), _op='tanh')


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    # Build a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)

    return nodes, edges


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='07_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(
            name=uid,
            label=f'{{ {n.label} | data: {n.data:.4f} | grad: {n.grad:.4f}}}',
            shape='record'
        )
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def main() -> None:
    # inputs
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias
    b = Value(6.8813735870195432, label='b')

    # pass through perceptron
    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
    logit = x1w1x2w2 + b; logit.label = 'logit'

    # pass through activation
    L = logit.tanh()
    L.label = 'L'

    # manual backpropagation
    # L = 1 * L
    # dL/dL = 1
    L.grad = 1.0

    # L = tanh(logit)
    # dL/dlogit = 1 - tanh(logit)**2
    logit.grad = 1.0 - L.data**2

    # logit = x1w1+x2w2 + b
    # dlogit/db = 1
    # dlogit/dx1w1+x2w2 = 1
    b.grad = 0.5
    x1w1x2w2.grad = 0.5

    # x1w1+x2w2 = x1w1 + x2w2
    # dx1w1+x2w2/dx1w1 = 1
    # dx1w1+x2w2/dx2w2 = 1
    x1w1.grad = 0.5
    x2w2.grad = 0.5

    # dL/dw1 = dx1w1+x2w2/dx1w1 * dx1w1/w1
    # dL/dx1 = dx1w1+x2w2/dx1w1 * dx1w1/x1
    # dL/dw2 = dx1w1+x2w2/dx2w2 * dx2w2/w2
    # dL/dx2 = dx1w1+x2w2/dx2w2 * dx2w2/x2
    # x1w1 = x1 * w1
    # dx1w1/w1 = x1
    # dx1w1/x1 = w1
    w1.grad = x1w1.grad * x1.data
    x1.grad = x1w1.grad * w1.data
    w2.grad = x2w2.grad * x2.data
    x2.grad = x2w2.grad * w2.data

    draw_dot(L).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
