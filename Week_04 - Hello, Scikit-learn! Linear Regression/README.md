# In class

Open and solve all tasks in `1_simple_linear_regression_with_numpy.ipynb` and `2_linear_regression_with_sklearn.ipynb`.

# For home

## Task 1

**Description:**

Create a machine learning pipeline to perform linear regression and predict the price of a used car. Use [this Kaggle dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data).

**Acceptance criteria:**

1. Submit one Jupyter notebook (the `csv` file is not required).
2. Perform Exploratory Data Analysis - minimum 5 figures/plots. Investigate the relationship between the features.
3. Perform Exploratory Data Analysis - minimum 10 tables - practice grouping, aggregating, obtaining statistics for various columns.
4. Perform Feature Engineering - create at least 1 new feature.
5. Train and evaluate a linear regression model - calculate `MAE`, `MSE`, `RMSE`.
6. Interpret at least three of the obtained coefficients.
7. `numpy` is used wherever possible.

## Task 2

**Description:**

Create a `Mat` class to represent a matrix and add the following functionality to it:

- constructor that takes number of rows and number of columns and allocates a matrix with that size filled with zeros;
- a method `set_at`: takes an index of row, an index of a column and a value and sets the value at that position in the matrix to the passed value;
- a method `randomize`: takes starting and ending values and assigns random values to the cells of the matrix using that interval;
- the `__repr__` method is predefined to be able to print the matrix.

**Expected behavior:**

Input

```python
np.random.seed(42)

inp = Mat(1, 2)
w1 = Mat(2, 2)
b1 = Mat(1, 2)
w2 = Mat(2, 1)
b2 = Mat(1, 1)

inp.set_at(0, 0, 0)
inp.set_at(0, 1, 1)

w1.randomize(0, 1)
b1.randomize(0, 1)
w2.randomize(0, 1)
b2.randomize(0, 1)

print(inp)
print(w1)
print(b1)
print(w2)
print(b2)
```

Output:

```text
w1: [[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
inp: [[0. 1.]]
b1: [[0.15601864 0.15599452]]
w2: [[0.05808361]
 [0.86617615]]
b2: [[0.60111501]]
```

## Task 3

**Description:**

Create two functions that can add and multiply matrices - `mat_add` and `mat_mul`.

**Acceptance criteria:**

1. `numpy` is used wherever possible.

**Expected behavior:**

Input

```python
w1 = Mat(2, 2)
w2 = Mat(2, 2)

w1.set_at(0, 0, 1)
w1.set_at(0, 1, 2)
w1.set_at(1, 0, 3)
w1.set_at(1, 1, 4)

w2.set_at(0, 0, 5)
w2.set_at(0, 1, 6)
w2.set_at(1, 0, 7)
w2.set_at(1, 1, 8)

print(f'Result of addition: {mat_add(w1, w2)}')
print(f'Result of multiplication: {mat_mult(w1, w2)}')
```

Output

```text
Result of addition: [[ 6.  8.]
 [10. 12.]]
Result of multiplication: [[19. 22.]
 [43. 50.]]
```

## Task 4

**Description:**

Create one additional function - `mat_sig`, that applies the sigmoid function to every element in a matrix.

**Acceptance criteria:**

1. Create a test.
