"""First project objective: multi layer perceptron MVP."""

import nanotorch as nt
import nanotorch.nn as nn

N_SAMPLES = 100_000
N_FEATURES = 8
HIDDEN_SIZE = 64
N_EPOCH = 50
BATCH_SIZE = 32
LR = 1e-3


def main():
    X: nt.Tensor = nt.rand(N_SAMPLES, N_FEATURES, dtype=nt.float32)
    y = X.sum(axis=1) + 3.14
    model = nn.Sequential(
        nn.Linear(N_FEATURES, HIDDEN_SIZE, bias=True),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 1),
    )
    optimizer = nn.GradientDescent(model.parameters(), lr=LR)
    for epoch in range(N_EPOCH):
        for i in range(0, N_SAMPLES // BATCH_SIZE + 1, BATCH_SIZE):
            xb, yb_true = X[i : i + BATCH_SIZE], y[i : i + BATCH_SIZE]
            yb_pred = model(xb)
            loss = nt.mean((yb_pred - yb_true) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, {loss=}")


if __name__ == "__main__":
    main()
