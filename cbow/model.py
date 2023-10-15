import numpy as np

from nli_learning.Dataset.NLIDataModule import NLIDataModule
from nli_learning.cbow.bow_svc import (
    train_logreg,
    train_svc,
    train_xgb,
    transform_dataset_to_bow,
)
from sklearn.preprocessing import MinMaxScaler


def train_pipeline(args):
    dataset_path = "dataset/datasets"
    dataset = NLIDataModule(
        data_source_folder=dataset_path, dataset_name="bow_dataset", batch_size=64
    )
    dataset.prepare_data()
    dataset.setup()

    print("Read Dataset")

    train_dataset = dataset.train_dataloader()
    val_dataset = dataset.val_dataloader()
    test_dataset = dataset.test_dataloader()

    print("Transform Dataset")

    train_inputs, train_labels = transform_dataset_to_bow(train_dataset)
    val_inputs, val_labels = transform_dataset_to_bow(val_dataset)
    test_inputs, test_labels = transform_dataset_to_bow(test_dataset)

    scaler = MinMaxScaler()

    train_inputs = scaler.fit_transform(train_inputs)
    val_inputs = scaler.transform(val_inputs)
    test_inputs = scaler.transform(test_inputs)

    if args.subtype == "svc":
        print("Train SVC")
        train_svc(
            train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels
        )
    elif args.subtype == "xgb":
        print("Train XGB")
        train_xgb(
            train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels
        )
    elif args.subtype == "logreg":
        print("Train Logistic Regression")
        train_logreg(
            train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels
        )
    else:
        raise NotImplemented("Unsupported model type. Please use either svc or xgb.")
