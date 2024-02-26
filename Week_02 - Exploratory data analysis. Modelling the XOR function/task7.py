import numpy as np

def create_dataset_and() -> list[(int, int, int)]:
    return [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]


def create_dataset_or() -> list[(int, int, int)]:
    return [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]


def initialize_weights(x: int, y: int) -> float:
    return np.random.uniform(x, y)


def calculate_loss(w1: int, w2: int, bias: int,
                   dataset: list[(int, int, int)]) -> float:
    sum = 0
    n = len(dataset)
    for (x, y, expected) in dataset:
        actual = x * w1 + y * w2 + bias
        sum += (actual - expected)**2
    return sum / n


def main() -> None:
    and_dataset = create_dataset_and()
    w1 = initialize_weights(0, 10)
    w2 = initialize_weights(0, 10)
    bias = initialize_weights(0, 10)

    eps = 0.1
    learning_rate = 0.1
    epochs = 100_000

    for _ in range(epochs):
        loss = calculate_loss(w1, w2, bias, and_dataset)
        der_loss_w1 = (calculate_loss(w1 + eps, w2, bias, and_dataset) -
                       loss) / eps
        der_loss_w2 = (calculate_loss(w1, w2 + eps, bias, and_dataset) -
                       loss) / eps
        der_loss_bias = (calculate_loss(w1, w2, bias + eps, and_dataset) -
                         loss) / eps

        w1 -= learning_rate * der_loss_w1
        w2 -= learning_rate * der_loss_w2
        bias -= learning_rate * der_loss_bias

    print(
        f'loss now is: {calculate_loss(w1, w2, bias, and_dataset)} for and prediction'
    )
    print('and predictions:')
    for i in range(2):
        for j in range(2):
            print(i, j, w1 * i + w2 * j + bias)

    or_dataset = create_dataset_or()
    w1 = initialize_weights(0, 3)
    w2 = initialize_weights(0, 3)
    bias = initialize_weights(0, 3)

    for _ in range(epochs):
        loss = calculate_loss(w1, w2, bias, or_dataset)
        der_loss_w1 = (calculate_loss(w1 + eps, w2, bias, or_dataset) -
                       loss) / eps
        der_loss_w2 = (calculate_loss(w1, w2 + eps, bias, or_dataset) -
                       loss) / eps
        der_loss_bias = (calculate_loss(w1, w2, bias + eps, or_dataset) -
                         loss) / eps

        w1 -= learning_rate * der_loss_w1
        w2 -= learning_rate * der_loss_w2
        bias -= learning_rate * der_loss_bias

    print(
        f'loss now is: {calculate_loss(w1, w2, bias, or_dataset)} for or prediction'
    )
    print('or predictions:')
    for i in range(2):
        for j in range(2):
            print(i, j, w1 * i + w2 * j + bias)


if __name__ == '__main__':
    main()