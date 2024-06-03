from __future__ import annotations

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

    def __sub__(self, rhs: float) -> Value:
        return self + Value(-rhs, label=str(-rhs))

    def __radd__(self, rhs: float) -> Value:
        return self + Value(rhs, label=str(rhs))

    def __add__(self, rhs) -> Value:
        if isinstance(rhs, (float, int)):
            rhs = Value(rhs, label=str(rhs))

        def backward():
            self.grad += result.grad
            rhs.grad += result.grad
            self._backward()
            rhs._backward()

        result = Value(self.data + rhs.data, _children=(self, rhs), _op='+')
        result._backward = backward
        return result

    def __rmul__(self, rhs: float) -> Value:
        return self * Value(rhs, label=str(rhs))

    def __mul__(self, rhs: Value) -> Value:
        if isinstance(rhs, (float, int)):
            rhs = Value(rhs, label=str(rhs))

        def backward():
            self.grad += result.grad * rhs.data
            rhs.grad += result.grad * self.data
            self._backward()
            rhs._backward()

        result = Value(self.data * rhs.data, _children=(self, rhs), _op='*')
        result._backward = backward
        return result

    def __pow__(self, value: float | int) -> Value:
        return Value(self.data**value, _children=(self, ), _op='**')

    def __truediv__(self, rhs: Value) -> Value:
        return self * rhs**(-1)

    def tanh(self) -> Value:
        def backward():
            self.grad += result.grad * (1 - result.data**2)
            self._backward()

        data = (np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)
        result = Value(data=data, _children=(self, ), _op='tanh')
        result._backward = backward
        return result

    def exp(self) -> Value:
        return Value(np.exp(self.data), _children=(self, ), _op='e')

    def backward(self) -> None:
        self.grad = 1.0
        predecessors = top_sort(self)
        for predecessor in reversed(predecessors):
            predecessor._backward()

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
    x = Value(2.0, label='x')

    expected = Value(4.0)

    actuals = {
        'actual_sum_l': x + 2.0,
        'actual_sum_r': 2.0 + x,
        'actual_mul_l': x * 2.0,
        'actual_mul_r': 2.0 * x,
        'actual_div_r': (x + 6.0) / 2.0,
        'actual_pow_l': x**2,
        'actual_exp_e': x**2,
    }

    assert x.exp().data == np.exp(
        2), f"Mismatch for exponentiating Euler's number: expected {np.exp(2)}, but got {x.exp().data}."

    for actual_name, actual_value in actuals.items():
        assert actual_value.data == expected.data, f'Mismatch for {actual_name}: expected {expected.data}, but got {actual_value.data}.'

    print('All tests passed!')


if __name__ == '__main__':
    main()
