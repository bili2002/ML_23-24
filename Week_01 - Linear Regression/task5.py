import numpy as np

np.random.seed(42)


def create_dataset(n: int) -> list[tuple[int, int]]:
    return [(n, n * 2) for n in range(n + 1)]


def initialize_weights(x: int, y: int) -> float:
    return np.random.uniform(x, y)


def calculate_loss(w: int, dataset: list[tuple[int, int]]) -> float:
    loss = 0
    n = len(dataset)
    for (x, expected) in dataset:
        actual = w * x
        loss += (actual - expected)**2
    loss /= n
    return loss


def main() -> None:
    # y = w*x
    num_samples = 5
    dataset = create_dataset(num_samples)
    w = initialize_weights(0, 10)

    eps = 0.001
    learning_rate = 0.001
    num_epochs = 500

    for _ in range(num_epochs):
        loss = calculate_loss(w, dataset)
        approximate_derivative = (calculate_loss(
            w + eps, dataset) - loss) / eps
        w -= learning_rate * approximate_derivative
        print(f'{loss=} | {w=}')

    print(f'{w=}')


if __name__ == '__main__':
    main()
