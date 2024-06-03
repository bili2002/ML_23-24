from __future__ import annotations
import graphviz


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
    dot = graphviz.Digraph(filename='04_result', format='svg', graph_attr={
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


def manual_der():
    h = 0.001
    
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0); f.label = 'f'
    L = d * f; L.label = 'L'
    L1 = L.data
    
    # Uncomment one addition of h at a time to see derivative value
    a = Value(2.0, label='a') # ; a.data += h
    b = Value(-3.0, label='b') # ; b.data += h
    c = Value(10.0, label='c') # ; c.data += h
    e = a * b; e.label = 'e' # ; e.data += h
    d = e + c; d.label = 'd' # ; d.data += h
    f = Value(-2.0); f.label = 'f' # ; f.data += h
    L = d * f; L.label = 'L' # ; L.data += h
    L2 = L.data
    
    print((L2 - L1) / h)


def main() -> None:
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a * b
    e.label = 'e'

    d = e + c
    d.label = 'd'

    f = Value(-2.0)
    f.label = 'f'

    L = d * f
    L.label = 'L'
    
    # If we change L by a very small amount, how does L change?
    # L = 1 * L
    # dL/dL = d/dL = 1
    L.grad = 1.0
    
    # L = d * f
    # dL/dd = f
    # dL/df = d
    d.grad = f.data
    f.grad = d.data
    
    # d = e + c
    # dd/de = 1
    # dd/dc = 1
    # dL/de = dL/dd * dd/de = -2 * 1 = -2
    # dL/dc = dL/dd * dd/dc = -2 * 1 = -2
    e.grad = -2.0
    c.grad = -2.0
    
    # e = a * b
    # de/da = b
    # de/db = a
    # dL/da = dL/dd * dd/de * de/da = -2 * de/da = -2 * b = -2 * -3 = 6
    # dL/db = dL/dd * dd/de * de/db = -2 * de/db = -2 * a = -2 * 2 = -4
    a.grad = 6.0
    b.grad = -4.0
    
    draw_dot(L).render(directory='./graphviz_output', view=True)
    
    manual_der()


if __name__ == '__main__':
    main()
