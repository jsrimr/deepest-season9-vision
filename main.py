import argparse
import os
import random

import neptune
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from warmup_scheduler import GradualWarmupScheduler

neptune.init(project_qualified_name='jeffrey/deepest-season9',
             api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNDA0MjIyMmQtZmFkNC00NzlmLWJmNTctMjJmZTcwNDg4OTc5In0=",
             )

RESUME = False
EVALUATE = False
checkpoint_path = None

print(torch.__version__)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

parser = argparse.ArgumentParser(description='PyTorch garbage classification Training')

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--validation_ratio', default=0.1, type=float)
parser.add_argument('--random_seed', default=10, type=int)
parser.add_argument('--initial_lr', default=1e-2 * torch.cuda.device_count(), type=float)
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--alpha', default=1., type=float)
args = parser.parse_args()

batch_size = args.batch_size
validation_ratio = args.validation_ratio
random_seed = args.random_seed
initial_lr = args.initial_lr * torch.cuda.device_count()
num_epoch = args.num_epoch
alpha = args.alpha

from mobilenet import MobileNetV3
import numpy as np
from augmentation import rand_bbox


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        r = random.random()
        if r <= 0.25:  # mixup
            indices = torch.randperm(inputs.size(0))
            shuffled_data = inputs[indices]
            shuffled_target = labels[indices]

            lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.7)
            inputs = lam * inputs + (1 - lam) * shuffled_data

            outputs = model(inputs)
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, shuffled_target)

        elif 0.25 < r <= 0.5:  # cutmix
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
        # outputs = model(inputs)
        # loss = criterion(outputs, labels)

        # loss.backward()
        loss.mean().backward()
        optimizer.step()
        # lr_scheduler.step()
        # scheduler_warmup.step(epoch)
        neptune.log_metric("lr", optimizer.param_groups[0]['lr'])
        scheduler.step(epoch)
        print(f"epoch = {epoch + 1}, lr = {optimizer.param_groups[0]['lr']}")

        running_loss += loss.item()

        show_period = 1
        correct = 0
        total = 0
        if i % show_period == show_period - 1:  # print every "show_period" mini-batches
            print(f'[%d, %5d/{len(train_loader) * batch_size}] loss: %.4f' %
                  (epoch + 1, (i + 1) * batch_size, running_loss / show_period))

            # # train acc
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            # print('[%d epoch] Accuracy of the network on the train images: %d %%' %
            #       (epoch + 1, 100 * correct / total)
            #       )
        running_loss = 0.0
        return running_loss


def test(test_loader, model, epoch, ):
    # validation part
    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
          (epoch + 1, 100 * correct / total)
          )
    return correct / total


def main():
    # Data
    data_dir = 'garbage/Garbage/'
    classes = os.listdir(data_dir)
    print(classes)

    transform_train = transforms.Compose([
        # transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    trainset = ImageFolder(data_dir, transform=transform_train)

    testset = ImageFolder(data_dir, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(testset, batch_size, num_workers=4, pin_memory=True, shuffle=True)

    # Model
    # model = DenseNetBC_100_12()
    # python; cifar.py - a; densenet - -depth; 190 - -growthRate; 40 - -train - batch; 64 - -epochs; 300 - -schedule; 150; 225 - -wd; 1e-4 - -gamma; 0.1 - -checkpoint
    # model = densenet(num_classes=6, depth=190, growthRate=40)
    model = MobileNetV3(n_class=6)
    model = torch.nn.DataParallel(model).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=.9)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=[int(num_epoch * 0.5), int(num_epoch * 0.75)], gamma=0.1,
                                                  last_epoch=-1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=lr_scheduler)

    best_acc = 0
    if RESUME:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(checkpoint_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # if EVALUATE:
    #     print('\nEvaluation only')
    #     test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    #     print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    #     return

    # Train and val

    for epoch in range(num_epoch):
        # neptune.log_metric("num_epcohs", epoch)
        _ = train(train_loader, model, criterion, optimizer, scheduler_warmup, epoch)
        test_acc = test(test_loader, model, epoch)
        neptune.log_metric("test_acc", test_acc)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)


import shutil


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
    neptune.create_experiment(
        params=args.__dict__,
        tags=['MBv3'])
    main()
