# 🎯 Goals for week 13

1. Implement a `Value` class and visualize a computational graph.
2. See backpropagation in action.
3. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1

**Description:**

Create a class `Value` that stores a single floating point number and implements the output operator.

**Test cases:**

```python
def main() -> None:
    value1 = Value(5)
    print(value1)

    value2 = Value(6)
    print(value2)
```

should output

```console
Value(data=5)
Value(data=6)
```

## Task 2

**Description:**

Extend the `Value` class by implementing functionality to add two values.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    result = x + y
    print(result)
```

should output

```console
Value(data=-1.0)
```

## Task 3

**Description:**

Extend the `Value` class by implementing functionality to multiply two values.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result)
```

should output

```console
Value(data=4.0)
```

## Task 4

**Description:**

Extend the `Value` class with another state variable that holds the values that produced the current value.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._prev)
```

should output

```console
{Value(data=-6.0), Value(data=10.0)}
```

## Task 5

**Description:**

Extend the `Value` class with another state variable that holds the operation that produced the current value.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)
```

should output

```console
+
```

## Task 6

**Description:**

Implement a function that takes a `Value` object and returns the nodes and edges that lead to the passed object.

**Test case:**

```python
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
```

should output

```console
x
nodes={Value(data=2.0)}
edges=set()
y
nodes={Value(data=-3.0)}
edges=set()
z
nodes={Value(data=10.0)}
edges=set()
result
nodes={Value(data=10.0), Value(data=-3.0), Value(data=4.0), Value(data=-6.0), Value(data=2.0)}
edges={(Value(data=-6.0), Value(data=4.0)), (Value(data=10.0), Value(data=4.0)), (Value(data=-3.0), Value(data=-6.0)), (Value(data=2.0), Value(data=-6.0))}
```

## Task 7

**Description:**

Let's visualize the tree leading to a certain value. We'll be using the Python package [graphviz](https://pypi.org/project/graphviz/). Note, that before installing it, you should have graphviz installed. Graphviz is available for installation [here](https://graphviz.org/download/). After installing, run the command `pip install graphiviz`.

Add the following code and ensure the test case runs successfully. Note that if by now you've run all scripts from the integrated terminal in vscode, you should now run this script from a terminal/command prompt that is **not** in VSCode.

```python
def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ data: {n.data} }}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
```

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    output_directory = './graphviz_output'
    draw_dot(result).render(directory=output_directory, view=True)
```

should output

![01_result](graphviz_output/01_result.svg?raw=true "01_result.png")
