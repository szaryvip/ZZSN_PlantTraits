import torch
from torch.utils.data import Dataset


class PGLSDataset(Dataset):
    def __init__(
        self,
        tabular_data,
        image_folder,
        transform_csv=None,
        transform_train=None,
        transform_val=None,
    ):
        """
        Args:
            csv_file (string): Path to the CSV file containing data.
            transform (callable, optional): Optional transform
                to be applied on a sample.
        """
        self.data_frame = tabular_data
        self.image_folder = image_folder
        self.transform_csv = transform_csv
        self.targets = ["X4", "X11", "X18", "X26", "X50", "X3112"]
        self.transform_train = transform_train
        self.transform_val = transform_val

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_frame.iloc[idx]
        id = int(sample["id"])
        image = self.image_folder.loader(
            self.image_folder.root + "/0/" + str(id) + ".jpeg"
        )

        if self.image_folder.transform is not None:
            image = self.image_folder.transform(image)

        if self.transform_train is not None and self.is_train:
            image = self.transform_train(image)
        elif self.transform_val is not None and not self.is_train:
            image = self.transform_val(image)

        try:
            targets = sample[[f"{target}_mean" for target in self.targets]].values
        except KeyError:
            targets = None

        features = sample.drop(["id"])
        if targets is not None:
            features = features.drop(
                [f"{target}_mean" for target in self.targets]
                + [f"{target}_sd" for target in self.targets]
            )

        features = torch.tensor(features.values, dtype=torch.float32)
        if targets is not None:
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.tensor([0] * 6, dtype=torch.float32)

        if self.transform_csv:
            features = self.transform_csv(features)

        return id, image, features, targets
