# 🎯 Goals for week 14

1. Do manual backpropagation and implement automatic backpropagation.
2. Add the hyperbolic tangent as an activation function.
3. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1

**Description:**

Include a label of the node in the visualization shown by graphviz.

For this to be implemented, the `Value` class would need to be extended with another state variable that holds a label of the current object being created.

**Test case:**

```python
def main() -> None:
    a = Value(2.0, label='x')
    b = Value(-3.0, label='y')
    c = Value(10.0, label='z')

    e = a * b
    e.label = 'e'

    d = e + c
    d.label = 'd'

    draw_dot(d).render(directory='./graphviz_output', view=True)
```

should output

![02_result](graphviz_output/02_result.svg?raw=true "02_result.png")

## Task 2

**Description:**

Now we're going to start to back-propagating derivatives to see how slight nudges (changes) to each of the variables change the value of the final variable - `L`.

Extend the `Value` class by adding a new state variable that will hold the gradient (derivative value) of that value with respect to `L`. By default it should be initialized with `0`, meaning "no effect" (initially we're assuming that the values do not affect the output).

Visualize the gradient of each node via `graphviz`.

**Test case:**

```python
def main() -> None:
    a = Value(2.0, label='x')
    b = Value(-3.0, label='y')
    c = Value(10.0, label='z')

    e = a * b
    e.label = 'e'

    d = e + c
    d.label = 'd'

    draw_dot(d).render(directory='./graphviz_output', view=True)
```

should output

![03_result](graphviz_output/03_result.svg?raw=true "03_result.png")

## Task 3

**Description:**

Manually do backpropagation on the generated computation graph.

![03_result](graphviz_output/03_result.svg?raw=true "03_result.png")

**Acceptance criteria:**

1. The process through which the values for the gradients are calculated is shown in comments.
2. A function `manual_der` is defined that helps verify the calculations.

**Test case:**

Output should be:

![04_result](graphviz_output/04_result.svg?raw=true "04_result.png")

## Task 4

**Description:**

Increase all values in the direction of the gradient with `0.01`. Print the new value of the loss function.

**Test case:**

```console
>>> python3 task04.py
Old L = -8.0
New L = -7.286496
```

## Task 5

**Description:**

Implement a perceptron with two inputs (for now without an activation function). Name the output node (on which you're calling `draw_dot`) `logit` - this is the term for a value that has not been passed through an activation function.

Here's what the perceptron model looks like:

![neuron_model](assets/neuron_model.jpeg?raw=true "neuron_model.jpeg")

Use the following configuration:

```text
x1 = 2.0
x2 = 0.0
w1 = -3.0
w2 = 1.0
b = 6.7
```

**Test case:**

```console
>>> python3 task05.py
```

should produce the following output:

![05_result](graphviz_output/05_result.svg?raw=true "05_result.svg")

## Task 6

**Description:**

Add the hyperbolic tangent as an activation function.

Let's also change the value of the bias to be 6.8813735870195432 (so that we get derivative values with little numbers after the comma) and display the computational graph.

**Test case:**

```console
>>> python3 task06.py
```

should produce the following output:

![06_result](graphviz_output/06_result.svg?raw=true "06_result.svg")

## Task 7

**Description:**

Manually backpropagate the gradients.

**Test case:**

```console
>>> python3 task07.py
```

should produce the following output:

![07_result](graphviz_output/07_result.svg?raw=true "07_result.svg")

## Task 8

**Description:**

Codify the differentiation process so that it can be executed automatically using a `backward` method that is called on the final (right-most node).

To do this, we'll need to:

- add another field to the `Value` class called `_backward` for automatic differentiation of the addition operation;
- implement a function that accepts a list of `Value` objects and sort them topologically. You can use the following code:

```python
def top_sort(start: Value) -> list[Value]:
    result = []
    visited = set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            result.append(v)
    build(start)
    return result
```

- integrate `top_sort` in a method called `backward` (notice that the `_backward` functions will stay).

**Acceptance criteria:**

1. The gradient for addition is calculated automatically.

**Test case:**

```console
>>> python3 task08.py
```

should produce the following output:

![08_result](graphviz_output/08_result.svg?raw=true "08_result.svg")

## Task 9

**Description:**

Implement `_backward` for the multiplication and hyperbolic tangent operations.

Abstract away the need to set the leaf to `1.0` by implementing a `backward` method and calling it from `main()` instead.

**Acceptance criteria:**

1. The gradient for multiplication is calculated automatically.
2. The gradient for hyperbolic tangent application is calculated automatically.
3. The `backward` method is called from `main()`.

**Test case:**

```console
>>> python3 task09.py
```

should produce the following output:

![09_result](graphviz_output/09_result.svg?raw=true "09_result.svg")

## Task 10

**Description:**

Currently, when we use a variable more than once the gradient gets overwritten. It can be seen below that the gradient of `x` should be `2` because `y = 2 * x`, but it is instead `1`.

![10_result_bug](graphviz_output/10_result_bug.svg?raw=true "10_result_bug.svg")

To fix this, we can accumulate the gradient instead of resetting it every time `_backward` is called.

**Test case:**

```console
>>> python3 task10.py
```

should produce the following output:

![10_result](graphviz_output/10_result.svg?raw=true "10_result.svg")

## Task 11

**Description:**

Extend the value class to allow the following operations:

- adding a float to a `Value` object;
- multiplying a `Value` object with a float;
- dividing a `Value` object by a float;
- exponentiation of a `Value` object with a float;
- exponentiation of Euler's number with a `Value` object.

We'll add the backpropagation (i.e. the implementation of the `_backward` function in another task), so you needn't add it here.

**Test cases:**

```console
>>> python3 task11.py
```

should produce the following output:

```text
All tests passed!
```

## Task 12

**Description:**

Break down the hyperbolic tangent into the expressions that comprise it and backpropagate through them. Refer to the formula in [Wikipedia](https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions).

**Test case:**

```console
>>> python3 task12.py
```

should produce the following output:

![11_result](graphviz_output/11_result.svg?raw=true "11_result.svg")
