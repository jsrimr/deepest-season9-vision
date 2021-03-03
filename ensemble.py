import argparse
import os
import random
import shutil

import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import PROJECT_NAME, API_TOKEN
neptune.init(project_qualified_name=PROJECT_NAME,
             api_token=API_TOKEN,
             )

print(torch.__version__)
use_cuda = torch.cuda.is_available()
# use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

parser = argparse.ArgumentParser(description='PyTorch garbage classification Training')

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--validation_ratio', default=0.1, type=float)
parser.add_argument('--random_seed', default=10, type=int)
parser.add_argument('--initial_lr', default=1e-2, type=float)
parser.add_argument('--num_epoch', default=501, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--alpha', default=1., type=float)
parser.add_argument('--use_mix', dest='use_mix', action='store_true')
parser.add_argument('--no_use_mix', dest='use_mix', action='store_false')
parser.set_defaults(use_mix=True)

parser.add_argument('--save_dir', default="ensemble", type=str, help="where to save weights")
parser.add_argument('--checkpoint_path', default="'ensemble_checkpoint/checkpoint.pth.tar'", type=str, help="where to load weights")
parser.add_argument('--resume', default=False, type=bool)
args = parser.parse_args()

RESUME = args.resume

batch_size = args.batch_size
validation_ratio = args.validation_ratio
random_seed = args.random_seed
initial_lr = args.initial_lr * torch.cuda.device_count()
num_epoch = args.num_epoch
alpha = args.alpha
num_workers = args.num_workers

CYCLES = 5
epochs_per_cycle = num_epoch // CYCLES

from mobilenet import MobileNetV3
import numpy as np
from augmentation import rand_bbox
from utils import find_classes, make_dataset, AlbumentationsDataset, albumentations_transform, \
    albumentations_transform_test, proposed_lr, split_train_val


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    n_batch = len(train_loader)
    correct = 0
    total = 0
    total_loss = 0
    for i, data in enumerate(train_loader):
        lr = proposed_lr(initial_lr, i + n_batch * epoch, n_batch * epochs_per_cycle)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        neptune.log_metric("lr", lr)

        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.to(device)

        optimizer.zero_grad()

        if args.use_mix:
            r = random.random()
        else:
            r = 1
        if r <= 0.33:  # mixup
            indices = torch.randperm(inputs.size(0))
            shuffled_data = inputs[indices]
            shuffled_target = labels[indices]

            lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.7)
            inputs = lam * inputs + (1 - lam) * shuffled_data

            outputs = model(inputs)
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, shuffled_target)

        elif 0.33 < r <= 0.67:  # cutmix
            lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute outputs
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # loss.backward()
        loss.mean().backward()
        optimizer.step()
        # scheduler.step(epoch)
        print(f"epoch = {epoch + 1}, lr = {optimizer.param_groups[0]['lr']}")
        print(f'[%d epoch, %5d/{n_batch * batch_size}] loss: %.4f' %
              (epoch + 1, (i + 1) * batch_size, loss.item()))

    loss_mean = total_loss / n_batch
    acc = correct / total

    return loss_mean, acc


def test(test_loader, model, criterion, epoch, ):
    # validation part
    correct = 0
    total = 0
    total_loss = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
          (epoch + 1, 100 * correct / total)
          )

    loss_mean = total_loss / len(test_loader)
    acc = correct / total
    return loss_mean, acc


def main():
    # Data
    data_dir = 'garbage/Garbage/'
    classes, class_to_idx = find_classes(data_dir)

    paths, labels = make_dataset(data_dir, class_to_idx)
    print(classes)

    train_path, train_label, val_path, val_label = split_train_val(paths, labels)

    trainset = AlbumentationsDataset(
        file_paths=train_path,
        labels=train_label,
        transform=albumentations_transform,
    )
    valset = AlbumentationsDataset(
        file_paths=val_path,
        labels=val_label,
        transform=albumentations_transform_test,
    )

    train_loader = DataLoader(trainset, batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(valset, batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

    # Model
    model = MobileNetV3()
    model = torch.nn.DataParallel(model).to(device)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=.9)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    #                                               milestones=[int(num_epoch * 0.5), int(num_epoch * 0.75)], gamma=0.1,
    #                                               last_epoch=-1)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=lr_scheduler)

    if RESUME:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.checkpoint_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler_warmup.load_state_dict(checkpoint['optimizer'])

    # Train and val
    snapshots = []
    for model_num in range(CYCLES):
        best_acc = 0
        for epoch in range(epochs_per_cycle):
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, None, epoch)
            neptune.log_metric("train loss", train_loss)
            neptune.log_metric("train acc", train_acc)

            val_loss, val_acc = test(val_loader, model, criterion, epoch)
            neptune.log_metric("val_loss", val_loss)
            neptune.log_metric("val_acc", val_acc)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                # 'optimizer': scheduler_warmup.state_dict(),
            }, is_best, prefix=args.save_dir, model_num=model_num)
        snapshots.append(model.state_dict())

    # lowered batch_size to meet gpu limit
    test_loader = DataLoader(valset, 16, num_workers=num_workers, pin_memory=True, shuffle=True)
    se_acc = test_se(MobileNetV3, snapshots, 5, test_loader)
    neptune.log_metric("snapshot ensemble", se_acc)


def test_se(Model, weights, use_model_num, test_loader):
    index = len(weights) - use_model_num
    weights = weights[index:]
    model_list = [Model() for _ in weights]

    for model, weight in zip(model_list, weights):
        model = nn.DataParallel(model)
        model.load_state_dict(weight)
        model.eval()
        if use_cuda:
            model.cuda()

    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.float().cuda(), target.cuda()
        output_list = [model(data).unsqueeze(0) for model in model_list]
        output = torch.mean(torch.cat(output_list), 0).squeeze()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_acc = 100 * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), test_acc))

    return test_acc


def save_checkpoint(state, is_best, prefix='ensemble', filename='checkpoint.pth.tar', model_num=-1):
    filename = f"model_num{model_num}" + filename
    filepath = os.path.join(prefix, filename)
    if not os.path.exists(prefix):
        os.makedirs(prefix, exist_ok=True)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(prefix, f'model_num{model_num}_best.pth.tar'))


if __name__ == '__main__':
    neptune.create_experiment(
        params=args.__dict__,
        tags=['MBv3', 'mixup', 'cutmix', 'SE'])
    print(args.__dict__)
    main()
