import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from data import *
from sklearn.metrics import roc_auc_score


def split_test_set(test_mask):
    device = test_mask.device
    mask1, mask2 = split_dataset(test_mask.size(0), 0.5)
    mask1, mask2 = torch.from_numpy(mask1).to(device), torch.from_numpy(mask2).to(device)
    with torch.no_grad():
        test_mask1, test_mask2 = (test_mask*mask1.float()).bool(), (test_mask*mask2.float()).bool()
    return test_mask1, test_mask2


def send_to_device(model, features, labels, train_mask, test_mask, device):
    model.to(device)
    train_features = torch.from_numpy(features[train_mask]).float().to(device)
    test_features = torch.from_numpy(features[test_mask]).float().to(device)
    train_labels = torch.from_numpy(labels[train_mask]).long().to(device)
    test_labels = torch.from_numpy(labels[test_mask]).to(device)
    labels = torch.from_numpy(labels).long().to(device)
    features = torch.from_numpy(features).float().to(device)
    train_mask = torch.from_numpy(train_mask).bool().to(device)
    test_mask = torch.from_numpy(test_mask).bool().to(device)
    return features, train_features, test_features, train_labels, test_labels, labels, train_mask, test_mask


def train(model, features, labels, train_mask, test_mask, args):
    metric = args.metric
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    data = send_to_device(model, features, labels, train_mask, test_mask, device)
    features, train_features, test_features, train_labels, test_labels, labels, train_mask, test_mask = data
    val_mask, test_mask = split_test_set(test_mask)
    criterion = torch.nn.functional.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    n_train_samples = train_labels.shape[0]
    n_samples = labels.shape[0]
    bs = args.bs
    best_val_metric = 0
    best_val_metric_epoch = -1
    val_metrics, test_metrics = [], []
    for step in range(args.epoch):
        model.train()
        shuffled_index = np.random.permutation(n_train_samples)
        count = 0
        while count < n_train_samples:
            minibatch = torch.from_numpy(shuffled_index[count: min(count + bs, n_train_samples)]).long().to(device)
            count = count + bs
            prediction = model(train_features[minibatch])
            loss = criterion(prediction, train_labels[minibatch], reduction='mean')
            loss.backward()
            optimizer.step()

        model.eval()
        predictions = []
        with torch.no_grad():
            count = 0
            while count < n_samples:
                minibatch = torch.tensor(list(range(count, min(count + bs, n_samples)))).long().to(device)
                count = count + bs
                prediction = model(features[minibatch])
                predictions.append(prediction)
            predictions = torch.cat(predictions, dim=0)
            loss_total = criterion(predictions[train_mask], train_labels, reduction='sum')
        train_metric = compute_metric(predictions, labels, train_mask, metric=metric)
        val_metric = compute_metric(predictions, labels, val_mask, metric=metric)
        test_metric = compute_metric(predictions, labels, test_mask, metric=metric)
        val_metrics.append(val_metric)
        test_metrics.append(test_metric)
        logging.info('epoch %d best test %s: %.4f, train loss: %.4f, train %s: %.4f val %s: %.4f test %s: %.4f' %
                     (step, metric, test_metrics[best_val_metric_epoch], loss_total / n_train_samples, metric, train_metric,
                      metric, val_metric, metric, test_metric))
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_val_metric_epoch = step
    best_test_metric = test_metrics[best_val_metric_epoch]
    logging.info('final test %s: %.4f (epoch: %d, val %s: %.4f)' %
                (metric, best_test_metric, best_val_metric_epoch, metric, best_val_metric))
    return best_test_metric


def compute_metric(prediction, labels, mask, metric='acc'):
    with torch.no_grad():
        if metric == 'acc':
            correct_prediction = (torch.argmax(prediction, dim=1) == labels)
            result = ((correct_prediction.float() * mask.float()).sum() / mask.float().sum()).item()
        elif metric == 'auc':
            # compute auc:
            prediction = torch.nn.functional.softmax(prediction[mask.bool()], dim=-1)
            multi_class = 'ovr'
            if prediction.size(1) == 2:
                prediction = prediction[:, 1]
                multi_class = 'raise'
            result = roc_auc_score(labels[mask.bool()].cpu().numpy(), prediction.cpu().numpy(), multi_class=multi_class)
        else:
            raise NotImplementedError
    return result