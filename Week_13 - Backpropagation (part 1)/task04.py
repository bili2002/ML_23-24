from __future__ import annotations


class Value:
    def __init__(self, data: float, _children: tuple=()) -> None:
        self.data = data
        
        # Set for efficiency. Also the order of the children does not matter.
        self._prev = set(_children)

    def __repr__(self) -> str:
        return f'Value(data={self.data})'

    def __add__(self, rhs: Value) -> Value:
        return Value(self.data + rhs.data, _children=(self, rhs))

    def __mul__(self, rhs: Value) -> Value:
        return Value(self.data * rhs.data, _children=(self, rhs))


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._prev)


if __name__ == '__main__':
    main()
