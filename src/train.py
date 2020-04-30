import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np


def train(model, features, labels, train_mask, test_mask, args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    train_features = torch.from_numpy(features[train_mask]).float().to(device)
    test_features = torch.from_numpy(features[test_mask]).float().to(device)
    train_labels = torch.from_numpy(labels[train_mask]).long().to(device)
    test_labels = torch.from_numpy(features[test_mask]).to(device)
    epoch = args.epoch
    criterion = torch.nn.functional.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    n_train_samples = train_labels.shape[0]
    n_samples = labels.shape[0]
    bs = args.bs
    best_test_acc = 0
    best_test_acc_epoch = -1
    for step in range(epoch):
        model.train()
        shuffled_index = np.random.permutation(n_train_samples)
        count = 0
        while count < n_train_samples:
            minibatch = torch.from_numpy(shuffled_index[count: min(count + bs, n_train_samples)]).long().to(device)
            count = count + bs
            prediction = model(train_features[minibatch])
            loss = criterion(prediction, labels[minibatch], reduction='mean')
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
        train_acc = compute_acc(predictions[train_mask], train_labels)
        test_acc = compute_acc(predictions[test_mask], test_labels)
        logging.info('epoch %d best acc: %.4f, train loss: %.4f, train acc: %.4f  test acc: %.4f' % (
        step, best_test_acc, loss_total / n_train_samples, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_acc_epoch = step
    logging.info('final test acc: %.4f (epoch: %d)' % (best_test_acc, best_test_acc_epoch))
    return best_test_acc


def compute_acc(predictions, labels):
    correct_prediction = (torch.argmax(predictions, dim=1) == labels)
    acc = correct_prediction.float().sum().item() / len(labels)
    return acc