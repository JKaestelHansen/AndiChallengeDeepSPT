from torch import Tensor
import torch
from torch import nn
from torch.utils.data import Dataset


def train_epoch(model, optimizer, train_loader, device, dim):
    train_loss = 0
    train_acc = 0
    train_F1 = 0
    train_loss_states = 0
    train_loss_alpha = 0
    train_loss_loss_D = 0
    train_loss_jaccard = 0
    train_loss_TP_CP = 0
    train_count = 0
    for batch_idx, xb in enumerate(train_loader):
        x, y_states, y_alpha, y_D, y_CPs = xb
        batch_size = x.size(0)
        loss, acc, F1, loss_states, loss_alpha, loss_D, loss_jaccard, loss_TP_CP = train_batch(model, optimizer, xb, dim)

        train_loss += loss * batch_size
        train_acc += acc * batch_size
        train_F1 += F1 * batch_size
        train_loss_states += loss_states * batch_size
        train_loss_alpha += loss_alpha * batch_size
        train_loss_loss_D += loss_D * batch_size
        train_loss_jaccard += loss_jaccard * batch_size
        train_loss_TP_CP += loss_TP_CP * batch_size
        train_count += batch_size

    average_loss = train_loss / train_count
    average_acc = train_acc / train_count
    average_F1 = train_F1 / train_count
    average_loss_states = train_loss_states / train_count
    average_loss_alpha = train_loss_alpha / train_count
    average_loss_D = train_loss_loss_D / train_count
    average_loss_jaccard = train_loss_jaccard / train_count
    average_loss_TP_CP = train_loss_TP_CP / train_count

    return average_loss, average_acc.item(), average_F1, average_loss_states, average_loss_alpha, average_loss_D, average_loss_jaccard, average_loss_TP_CP


def train_batch(model, optimizer, xb, dim):
    model.train()
    optimizer.zero_grad()

    loss, acc, p, F1, loss_states, loss_alpha, loss_D, loss_jaccard, loss_TP_CP = model(xb, dim=dim)
    loss.backward()
    optimizer.step()

    return loss.item(), acc, F1, loss_states.item(), loss_alpha.item(), loss_D.item(), loss_jaccard, loss_TP_CP.item()


def validate(model, optimizer, validation_loader, device, dim):
    model.eval()

    val_loss = 0
    val_acc = 0
    val_F1 = 0
    val_loss_states = 0
    val_loss_alpha = 0
    val_loss_D = 0
    val_loss_jaccard = 0
    val_loss_TP_CP = 0
    val_count = 0
    with torch.no_grad():
        for batch_idx, xb in enumerate(validation_loader):
            x, y_states, y_alpha, y_D, y_CPs = xb
            batch_size = x.size(0)
            loss, acc, p, F1, loss_states, loss_alpha, loss_D, loss_jaccard, loss_TP_CP = model(xb, dim=dim)

            val_loss += loss.item() * batch_size
            val_acc += acc * batch_size
            val_F1 += F1 * batch_size
            val_loss_states += loss_states * batch_size
            val_loss_alpha += loss_alpha * batch_size
            val_loss_D += loss_D * batch_size
            val_loss_jaccard += loss_jaccard * batch_size
            val_loss_TP_CP += loss_TP_CP * batch_size
            val_count += batch_size

        average_loss = val_loss / val_count
        average_acc = val_acc / val_count
        average_F1 = val_F1 / val_count
        average_loss_states = val_loss_states / val_count
        average_loss_alpha = val_loss_alpha / val_count
        average_loss_D = val_loss_D / val_count
        average_loss_jaccard = val_loss_jaccard / val_count
        average_loss_TP_CP = val_loss_TP_CP / val_count

    return average_loss, average_acc.item(), average_F1, average_loss_states.item(), average_loss_alpha.item(), average_loss_D.item(), average_loss_jaccard, average_loss_TP_CP.item()


def check_pred(model, data_loader, ypadtoken=10):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in data_loader:
            _, _, pred = model(xb)
            preds.append([p[p!=ypadtoken] for p in pred])
    return preds