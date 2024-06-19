from __future__ import annotations
import graphviz

import numpy as np


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label='') -> None:
        self.grad = 0.0
        self.data = data
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Value(data={self.data})'

    def __add__(self, rhs: Value) -> Value:
        def backward():
            self.grad = result.grad
            rhs.grad = result.grad

        result = Value(self.data + rhs.data, _children=(self, rhs), _op='+')
        result._backward = backward
        return result

    def __mul__(self, rhs: Value) -> Value:
        return Value(self.data * rhs.data, _children=(self, rhs), _op='*')

    def tanh(self) -> Value:
        data = (np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)
        return Value(data=data, _children=(self, ), _op='tanh')

    def backward(self) -> None:
        self.grad = 1
        predecessors = top_sort(self)
        for predecessor in reversed(predecessors):
            predecessor._backward()


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
    dot = graphviz.Digraph(filename='08_result', format='svg', graph_attr={
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


def top_sort(start: Value) -> list[Value]:
    result = []
    visited = set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            # "start" value will be added after all of its children
            # i.e. it will be at the end of the list
            result.append(v)
    build(start)
    return result


def main() -> None:
    x = Value(5, label='x')
    y = Value(10, label='y')
    z = x + y; z.label = 'z'

    z.grad = 1
    z.backward()

    draw_dot(z).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
