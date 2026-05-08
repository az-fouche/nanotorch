"""End-to-end MLP fitting with CUDA support."""

import time

import nanotorch as nt
import nanotorch.nn as nn

N_SAMPLES = 100_000
N_FEATURES = 8
HIDDEN_SIZE = 256
N_EPOCH = 50
BATCH_SIZE = 1024
LR = 1e-4

device = "cpu"  # "cuda" if nt.is_cuda_available() else "cpu"


def main():
    X: nt.Tensor = nt.rand(N_SAMPLES, N_FEATURES, dtype=nt.float32).to(device)
    y = (X.sum(axis=1) + 3.14).to(device)
    model = nn.Sequential(
        nn.Linear(N_FEATURES, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 1),
    ).to(device)
    optimizer = nn.GradientDescent(model.parameters(), lr=LR)
    print(f"== Starting Training on {device} ==")
    for epoch in range(N_EPOCH):
        t_start = time.time()
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            xb, yb_true = (X[i : i + BATCH_SIZE], y[i : i + BATCH_SIZE])
            yb_pred = model(xb)
            loss = nt.mean((yb_pred.squeeze() - yb_true) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        optimizer._lr = max(1e-4, optimizer._lr * 0.95)
        samples_s = N_SAMPLES / (time.time() - t_start)
        print(f"Epoch {epoch}, loss={loss.cpu().item():.3f}, {samples_s:.1f} samples/s")

    print("== Training finished ==")
    for i in range(min(10, BATCH_SIZE)):
        print(f"true: {yb_true[i].cpu().item()}, pred:{yb_pred[i].cpu().item()}")


if __name__ == "__main__":
    main()
