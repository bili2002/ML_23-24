from __future__ import annotations


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

    print(f'Old L = {L.data}')

    # learning rate
    h = 0.01

    # gradient step
    a.data += h * a.grad
    b.data += h * b.grad
    c.data += h * c.grad
    f.data += h * f.grad

    # forward pass
    e = a * b
    d = e + c
    L = d * f

    print(f'New L = {L.data}')


if __name__ == '__main__':
    main()
