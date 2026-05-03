"""First project objective: multi layer perceptron MVP."""

import nanotorch as nt
import nanotorch.nn as nn

N_SAMPLES = 50_000
N_FEATURES = 8
HIDDEN_SIZE = 16
N_EPOCH = 20
BATCH_SIZE = 32
LR = 1e-4


def main():
    X: nt.Tensor = nt.rand(N_SAMPLES, N_FEATURES, dtype=nt.float32)
    y = X.sum(axis=1) + 3.14
    model = nn.Sequential(
        nn.Linear(N_FEATURES, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 1),
    )
    optimizer = nn.GradientDescent(model.parameters(), lr=LR)
    for epoch in range(N_EPOCH):
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            xb, yb_true = X[i : i + BATCH_SIZE], y[i : i + BATCH_SIZE]
            yb_pred = model(xb)
            loss = nt.mean((yb_pred - yb_true) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        optimizer._lr = max(1e-4, optimizer._lr * 0.95)
        print(f"Epoch {epoch}, {loss=}")

    for i in range(min(10, BATCH_SIZE)):
        print(f"true: {yb_true[i].item()}, pred:{yb_pred[i].item()}")


if __name__ == "__main__":
    main()
