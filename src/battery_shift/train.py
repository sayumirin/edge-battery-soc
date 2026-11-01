import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit(model, dataset, cfg, validation=None, mode="all"):
    model = copy.deepcopy(model)
    for name, p in model.named_parameters():
        p.requires_grad_(mode == "all" or name.startswith(mode + "."))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    best, best_loss, stale, history = None, float("inf"), 0, []
    # Adaptation can fix the number of optimizer steps, independent of the adaptation-set size.
    budget = cfg.get("adapt_steps") if validation is None else None
    if budget:
        epochs = -(-budget // len(loader))
    else:
        epochs = cfg.epochs if validation is not None else cfg.adapt_epochs
    steps = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in loader:
            if budget and steps == budget:
                break
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            steps += 1
            losses.append(loss.item())
        score = float(np.mean(losses))
        if validation is not None:
            from .evaluate import predict

            y, p = predict(model, validation, cfg.batch_size)
            score = float(np.mean((p - y) ** 2))
        history.append(dict(epoch=epoch + 1, steps=steps, loss=float(np.mean(losses)), selection_mse=score))
        if validation is None or score < best_loss:
            best, best_loss, stale = copy.deepcopy(model.state_dict()), score, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best)
    return model.eval(), history
