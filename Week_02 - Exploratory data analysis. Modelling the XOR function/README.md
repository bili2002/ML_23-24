# Goals for week 02

1. Introduce `pandas`.
2. Introduce `matplotlib`.
3. Modelling the `XOR` function.

## Task 1

Solve all tasks from `pandas_part1.ipynb` - simple exploratory data analysis, part 1.

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 2

Solve all tasks from `pandas_part2.ipynb` - simple exploratory data analysis, part 2.

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 3

Solve all tasks from `pandas_part3.ipynb` - filtering and sorting, part 1.

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 4

Solve all tasks from `pandas_part4.ipynb` - filtering and sorting, part 2.

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 5

Solve all tasks from `pandas_part5.ipynb` - filtering and sorting, part 3.

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 6

Solve all tasks from `pandas_part6.ipynb` - grouping.

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## For home

### Task 1

**Description:**

Answer the following questions:

1. What are features?
2. What are observations?
3. What does feature engineering mean?
4. What does feature selection mean?
5. What are hyperparameters?
6. Give three examples of hyperparameters?
7. What does it mean for a model to be trained for 50 epochs?
8. What happens at the end of each epoch?
9. Why should model selection only happen based on the test set?
10. Why should evaluation on the test set be performed only once?

**Acceptance criteria:**

1. Answers are written **in Bulgarian**.
2. Answers are written in the body of the email (not in a `.docx` or `.txt` file).

### Task 2

**Description:**

Solve all tasks from `for_home_pandas.ipynb`. Refer to `hello_pandas.ipynb` for examples.

**Acceptance criteria**

1. Real output matches expected output on tasks.

### Task 3

**Description:**

Go through `hello_matplotlib.ipynb` and understand the code and how it produces the output. After that solve all tasks from `for_home_matplotlib.ipynb`.

**Acceptance criteria**

1. Real output is close to the expected output.

### Task 4

**Description**

Add a sigmoid calculation to the code in `task7.py`. Plot the values it takes from `-10` till `10` using `matplotlib`.

**Acceptance criteria**

1. The above functionality is implemented.
2. A plot is produced using `matplotlib`.

**Expected behavior**

```python
def main() -> None:
    for i in range(-10, 11):
        print(f'{i} => {sigmoid(i)}')
```

```text
-10 => 4.5397868702434395e-05
-9 => 0.00012339457598623172
-8 => 0.0003353501304664781 
-7 => 0.0009110511944006454 
-6 => 0.0024726231566347743 
-5 => 0.0066928509242848554 
-4 => 0.01798620996209156   
-3 => 0.04742587317756678   
-2 => 0.11920292202211755   
-1 => 0.2689414213699951    
0 => 0.5
1 => 0.7310585786300049     
2 => 0.8807970779778823     
3 => 0.9525741268224334     
4 => 0.9820137900379085     
5 => 0.9933071490757153     
6 => 0.9975273768433653
7 => 0.9990889488055994
8 => 0.9996646498695336
9 => 0.9998766054240137
10 => 0.9999546021312976
```

### Task 5

**Description**

Use the sigmoid function during model training and inference. What happens to the value of the loss function?

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.
2. A comparison is made before and after applying sigmoid.

### Task 6

**Description**

Using `matplotlib` plot how the loss changes with epochs. Plot the epoch on the x-axis and the loss on the y-axis.

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

### Task 7

**Description**

Create a dataset and a model for the `NAND` logic gate. Can you reuse the already created models for the `AND` and `OR` gates and just change their datasets?

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

### Task 8

**Description**

Create a Python class named `Xor` and store all parameters and biases of the `Xor` model. Train and evaluate the model using an appropriate architecture.

**Acceptance criteria**

1. High quality code is written:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.
2. A `forward` function is implemented that takes a model, two inputs and returns the output of the model.
