import matplotlib.pyplot as plt
from utils import split_train_val, find_classes, make_dataset, AlbumentationsDataset, albumentations_transform, albumentations_transform_test, show_sample

def test_data_load():
    data_dir = 'garbage/Garbage/'
    classes, class_to_idx = find_classes(data_dir)

    paths, labels = make_dataset(data_dir, class_to_idx)
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

    show_sample(trainset[0][0])
    show_sample(valset[0][0])

if __name__ == '__main__':
    test_data_load()