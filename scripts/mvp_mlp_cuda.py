"""End-to-end MLP fitting with CUDA support."""

import argparse
import time

import tqdm

import nanotorch as nt
import nanotorch.nn as nn

N_SAMPLES = 200_000
N_FEATURES = 32
HIDDEN_SIZE = 256
N_EPOCH = 50
BATCH_SIZE = 1024
LR = 5e-4


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    args = arg_parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if nt.is_cuda_available() else "cpu"
    if device == "cuda" and not nt.is_cuda_available():
        raise RuntimeError("No CUDA device detected.")

    X: nt.Tensor = nt.rand(N_SAMPLES, N_FEATURES).to(device)
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
    for _ in range(N_EPOCH):
        pbar = tqdm.tqdm(range(0, N_SAMPLES, BATCH_SIZE), desc="Batch", ncols=80)
        for i in pbar:
            xb, yb_true = (X[i : i + BATCH_SIZE], y[i : i + BATCH_SIZE])
            yb_pred = model(xb)
            loss = nt.mean((yb_pred.squeeze() - yb_true) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.cpu().item():.3f}")
        optimizer._lr = max(1e-4, optimizer._lr * 0.95)

    print("== Training finished ==")
    for i in range(min(10, BATCH_SIZE)):
        print(f"true: {yb_true[i].cpu().item()}, pred:{yb_pred[i].cpu().item()}")


if __name__ == "__main__":
    main()
