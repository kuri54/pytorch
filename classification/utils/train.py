import torch
from sklearn.metrics import *

from utils.evaluator import *

# Trainループ
def train_loop(model, dataloaders, device, criterion, optimizer, scheduler):
    model.train()

    train_running_loss = 0.0
    train_running_corrects = 0

    for idx, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_running_loss += loss.item() * inputs.size(0)
        train_running_corrects += torch.sum(preds == labels.data)

    train_loss = train_running_loss / len(dataloaders.dataset)
    train_acc = train_running_corrects.double() / len(dataloaders.dataset)

    return train_loss, train_acc

# Validループ
def valid_loop(model, dataloaders, device, criterion):
    model.eval()

    valid_running_loss = 0.0
    valid_running_corrects = 0

    labels_all = []
    pred_all = []

    for idx, (inputs, labels) in enumerate(dataloaders):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            valid_running_loss += loss.item() * inputs.size(0)
            valid_running_corrects += torch.sum(preds == labels.data)

            pred_all.extend(predict.item() for predict in preds)
            labels_all.extend(label.item() for label in labels)

    valid_loss = valid_running_loss / len(dataloaders.dataset)
    valid_acc = valid_running_corrects.double() / len(dataloaders.dataset)

    return valid_loss, valid_acc, labels_all, pred_all

# test画像で検証 -> 可視化
def evalute_model(model, dataloaders, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        labels_all = []
        pred_all = []

        for idx, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)

            pred_all.extend(predict.item() for predict in preds)
            labels_all.extend(label.item() for label in labels)

    return correct, total, labels_all, pred_all
