#!/usr/bin/env python
from predict_eval import compute_eval
import train_torch

if __name__ == '__main__':
    train_torch.train()
    compute_eval()


