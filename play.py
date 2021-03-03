import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# # import neptune
#
# # neptune.init(project_qualified_name='jeffrey/deepest-season9',
#              api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNDA0MjIyMmQtZmFkNC00NzlmLWJmNTctMjJmZTcwNDg4OTc5In0=",
#              )

print(torch.__version__)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

parser = argparse.ArgumentParser(description='PyTorch garbage classification Training')

parser.add_argument('--save_dir', default="ensemble_final", type=str)
parser.add_argument('--ensemble', dest="single", action='store_false')
parser.add_argument('--single', dest="single", action='store_true')
parser.set_defaults(single=True)
args = parser.parse_args()

save_dir = args.save_dir

CYCLES = 5

from mobilenet import MobileNetV3
from utils import find_classes, make_dataset, AlbumentationsDataset, albumentations_transform_test


def test(test_loader, model, epoch, ):
    # validation part
    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.to(device)
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
    classes, class_to_idx = find_classes(data_dir)

    paths, labels = make_dataset(data_dir, class_to_idx)
    print(classes)

    testset = AlbumentationsDataset(
        file_paths=paths,
        labels=labels,
        transform=albumentations_transform_test,
    )

    test_loader = DataLoader(testset, 4, num_workers=0, pin_memory=True,
                             shuffle=True)  # sampler=test_sampler,

    # Model
    model = MobileNetV3()
    model = torch.nn.DataParallel(model).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    import glob
    snapshots = []
    for tar_file in glob.glob(f"{save_dir}/*best*.tar"):
        checkpoint = torch.load(tar_file)
        if args.single:
            if tar_file.endswith("model_num4_best.pth.tar"):
                print(tar_file)
                snapshots.append(checkpoint['state_dict'])
        else:
            print(tar_file)
            snapshots.append(checkpoint['state_dict'])
    print('\nEvaluation only')

    if args.single:
        se_acc = test_se(MobileNetV3, snapshots, 1, test_loader)
    else:
        se_acc = test_se(MobileNetV3, snapshots, 5, test_loader)
    print('Test Acc:  %.2f' % (se_acc))
    return
    # neptune.log_metric("snapshot ensemble", se_acc)


def test_se(Model, weights, use_model_num, test_loader):
    # device = "cpu"

    index = len(weights) - use_model_num
    weights = weights[index:]
    model_list = [Model() for _ in weights]

    for model, weight in zip(model_list, weights):
        model = nn.DataParallel(model)
        model.load_state_dict(weight)
        model.eval()
        model.to(device)
        print("loaded model")

    # for inputs, labels in test_loader:
    #     inputs, labels = inputs.float().to(device), labels.to(device)
    #     break

    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.to(device)

        if args.single:
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
        else:
            output_list = [model(inputs) for model in model_list]
            output = torch.mean(torch.stack(output_list), 0)
            pred = output.data.max(1)[1]

        correct += pred.eq(labels.data).cpu().sum()
        total += inputs.size(0)

    test_acc = correct / total * 100
    print(test_acc.item())
    return test_acc.item()


if __name__ == '__main__':
    #     # neptune.create_experiment(
    #     params=args.__dict__,
    #     tags=['MBv3', 'mixup', 'cutmix', 'SE'])
    print(f"args.single = {args.single}")
    main()
