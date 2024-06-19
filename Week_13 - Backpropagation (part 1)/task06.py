from __future__ import annotations


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = '') -> None:
        self.data = data
        self._prev = set(_children)
        self._op = _op

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


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z

    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')


if __name__ == '__main__':
    main()
