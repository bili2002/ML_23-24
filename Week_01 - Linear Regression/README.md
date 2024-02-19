# Task 1 - Introduction

**Description**

Introduce foundational concepts and create a notebook that prints "Hello, world!".

**Acceptance criteria**

1. Students to confirm they understand what machine learning is.
2. Students to confirm they understand what model training is.
3. Students to confirm they understand what the main roles in the field are.
4. Students to confirm they understand what task Large Language Models solve.
5. A jupyter notebook is used. Students to verify they have the proper setup and extensions.
6. Python virtual environments are explained.
7. A new Python virtual environment is created.
8. Code is added that prints "Hello, world!".
9. The purpose of `if __name__ == '__main__':` in Python scripts is explained.
10. `numpy` is introduced via `hello_numpy.ipynb`.
11. The purpose of metrics is explained.
12. Explain mean absolute error.
13. Explain mean squared error.
14. Why can't we cube the difference to get the error instead?
15. Relationship between derivatives and graphs of loss functions.
16. How can derivatives be approximated using the method of `finite differences`?

# Task 2 - Linear regression: Dataset creation and Parameter initialization

**Description**

We want to create and train a machine learning model that given `x` predicts `2 * x`.

Create the following functions:

1. `create_dataset`: accepts `n` and returns `n` consecutive samples that demonstrate the expected behavior.
2. `initialize_weights`: accepts `x` and `y` and returns a random number in the range [x, y).

**Acceptance criteria**

1. A Python file is used for the solution.
2. A general form of the model is placed in a comment that shows how many parameters the model has.
3. `numpy` is used to initialize the parameter(s) of the model.

**Test case**

```python
create_dataset(4)          # [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]
initialize_weights(0, 100) # 95.07143064099162
initialize_weights(0, 10)  # 3.745401188473625
```

# Task 3 - Linear regression: Measure how bad the model is

**Description**

We want to create and train a machine learning model that given `x` predicts `2 * x`.

Create a dataset with 5 samples.
Create a model that has a random value for its parameter in the interval [0, 10) with a seed set to `42`.

Create a function `calculate_loss` that accepts the model, a dataset and computes the mean squared error of the model over the dataset.

**Acceptance criteria**

1. The above functionality is implemented.
2. Experiment with the first parameter of the model - what happens to loss function when you pass `w + 0.001 * 2`, `w + 0.001`, `w - 0.001` and `w - 0.001 * 2`?

**Test case**

```python
loss = calculate_loss(w, dataset)
<print the loss> # MSE: 27.92556532998047
```

# Task 4 - Linear regression: Learning rate and finite differences

**Description**

We want to automatically update the value of the parameter using the method of finite differences to approximate the derivative. Approximate the derivative and subtract the resulting value from the parameter.

**Acceptance criteria**

1. The above functionality is implemented.
2. Print the value of the loss function before updating the parameter.
3. Print the value of the loss function after updating the parameter.
4. See how adding a `learning rate` impacts the value of the loss function.
5. Include a `for-loop` in which the process is repeated `10` times (also known as `epochs` - the number of times the whole dataset is traversed).

**Test cases**

```python
approximate_derivative = <calculation>
print(calculate_loss(w, dataset))
w -= approximate_derivative
print(calculate_loss(w, dataset))
```

compare it with

```python
learning_rate = 0.001
approximate_derivative = <calculation>
print(calculate_loss(w, dataset))
w -= learning_rate * approximate_derivative
print(calculate_loss(w, dataset))
```

# Task 5 - Training the model

**Description**

Train the model for `500 epochs` and print the value of `w`.

Experiment with removing the `seed` and seeing whether the training still converges.

Congratulations! You just created and trained your first neural network 🎉!

**Acceptance criteria**

1. The above functionality is implemented.

# For home

## Task 1

**Description**

Answer the following questions:

1. What is machine learning in 1 sentence?
2. What is does model training mean in 1 sentence?
3. What task do Large Language Models solve?
4. Why should we be using different virtual environments for different projects?
5. What is the difference between loss functions and metrics?
6. Why is the use of mean squared error preferred to the use of mean absolute error?
7. Why can't we cube the difference to get the mean error instead of squaring it?
8. Why is the derivative of the loss function beneficial during model training?

**Acceptance criteria**

1. Answers are written **in Bulgarian**.
2. Answers are written in the body of the email (not in a `.docx` or `.txt` file).

## Task 2

**Description**

Watch [this](https://www.youtube.com/watch?v=2kSl0xkq2lM) video and answer the following questions:

1. Which class of AI techniques began to work around 2005 for solving real-world problems?
2. What is a classic application of AI is discussed in the video?
3. What is the simplest way of getting Machine Learning to solve a task?
4. Define what supervised learning is.
5. What does AI require in order to work at all?
6. What class of tasks does the task for facial recognition belong to?
7. What is the task solved by each neuron in the brain?
8. What happens when a neuron recognizes a pattern?
9. What is a digital picture made up of?
10. What are the three reasons that allowed the modelling of human brain cells in software?
11. Which type of computer processor is suited really well for performing the mathematics related to deep learning?
12. Do the capabilities of neural networks grow with scale?
13. What allowed for the biggest and fastest advances in the field?
14. What is the name of the probably most important paper in the last decade?
15. What is the Transformer architecture designed for?
16. What is the innovation that the Transformer architecture has?
17. What is the key point about GPT-3?
18. Is machine learning more efficient at learning than humans? Why?
19. What is "the bitter truth"?
20. What is one reason why ChatGPT is not conscious?

**Acceptance criteria**

1. Answers are written in Bulgarian.
2. Answers are written in the body of the email (not in a `.docx` or `.txt` file).

## Task 3

**Description**

Solve and submit the tasks in the [Python advanced Jupyter notebook](../Week_00%20-%20Hello,%20Python/3_python_advanced.ipynb).

**Acceptance criteria**

1. Real output matches expected output.

## Task 4

Solve all tasks from `numpy_for_home.ipynb`.

**Acceptance criteria**

1. Real output matches expected output on tasks that do not include working with random numbers.

## Task 5

**Description**

In class we created a neural network with one neuron that had one input. In this task we're going to **model the `AND` and `OR`** circuit operations, thus the network is going to have two inputs and thus two weights attached to them. Create two models and train for 1,000 epochs and after each epoch print the values for the parameters of the models and the value of the loss function. When the two models are trained, apply each of them to their corresponding training sets and print the values they predict.

**Acceptance criteria**

1. A general form of the two models is placed in a comment that shows how many parameters they have.
2. A dataset of tuples with three elements is created for the `OR` gate.
3. A dataset of tuples with three elements is created for the `AND` gate.
4. The loss function is modified accordingly.
5. Weights are initialized randomly without seed.
6. The loss function decreases as more epochs are done.
7. The two models are trained for 1,000 epochs.
8. Print the values the models predict. What do you notice about them?

## Task 6

**Description**

Extend the functionality of task 5 (the previous task) by adding a free-floating parameter (i.e. a parameter that has a weight of `1`) - the `bias`. This would help the model drive the loss down to 0. Our models will now have three parameters, though, note that the `bias` is a parameter of the model itself, it's not part of the dataset.

The `bias` parameter allows the model to shift it's prediction when all of its weights become 0.

**Acceptance criteria**

1. A general form of the two models is placed in a comment that shows how many parameters they have.
2. Print the values the models predict. What has changed in comparison to task 5?
