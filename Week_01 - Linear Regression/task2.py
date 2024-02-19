import numpy as np

np.random.seed(42)


def create_dataset(n: int) -> list[tuple[int, int]]:
    return [(n, n * 2) for n in range(n + 1)]


def initialize_weights(x: int, y: int) -> float:
    return np.random.uniform(x, y)

def main() -> None:
    # y = w*x
    training_data = create_dataset(4)
    w = initialize_weights(0, 10)
    
    print(f'{training_data=}')
    print(f'{w=}')


if __name__ == '__main__':
    main()
