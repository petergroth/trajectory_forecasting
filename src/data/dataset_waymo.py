import os
import os.path as osp
import torch
# from src.utils import parse_sequence
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
# import tensorflow as tf
from typing import Optional


class OneStepWaymoTrainDataset(InMemoryDataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        super(OneStepWaymoTrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = os.listdir("data/raw/waymo/train")
        names.sort()
        return names

    @property
    def processed_file_names(self):
        # # Determine number of files
        # n_graphs = 0
        # # Load all files and iterate through to determine length. Thanks TF
        # for i, raw_path in enumerate(self.raw_file_names):
        #     # Load file
        #     dataset = tf.data.TFRecordDataset(
        #         self.raw_dir + "/" + raw_path, compression_type=""
        #     )
        #     for seq_idx, data in enumerate(dataset.as_numpy_iterator()):
        #         n_graphs += 1

        # return [f"waymo_train_data_{i:04}.pt" for i in range(n_graphs)]
        return ["waymo_train_data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "data/raw/waymo/train")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "data/processed/waymo/train/onestep")

    def download(self):
        pass

    def process(self):
        # Node attributes
        key_values = [
            "x",
            "y",
            "z",
            "velocity_x",
            "velocity_y",
            "bbox_yaw",
            "vel_yaw",
            "width",
            "length",
            "height",
            "valid",
        ]

        # Dimensions
        n_nodes = 128
        n_features = len(key_values)
        n_steps = 91
        data_list = []

        # Move through raw files
        for i, raw_path in enumerate(self.raw_file_names):
            # Load file
            dataset = tf.data.TFRecordDataset(
                self.raw_dir + "/" + raw_path, compression_type=""
            )
            for seq_idx, data in enumerate(dataset.as_numpy_iterator()):
                # Parse sequence
                parsed = parse_sequence(data)
                # Allocate tensor for full value array
                feature_matrix = torch.zeros((n_nodes, n_steps, n_features))

                # Fill in all values
                for j, key in enumerate(key_values):
                    # Encode history
                    feature_matrix[:, :10, j] = torch.Tensor(
                        parsed["state/past/" + key].numpy()
                    )
                    feature_matrix[:, 10, j] = torch.Tensor(
                        parsed["state/current/" + key].numpy()
                    ).squeeze()
                    feature_matrix[:, 11:, j] = torch.Tensor(
                        parsed["state/future/" + key].numpy()
                    )

                # Determine input/target pairs
                for t in range(n_steps - 1):
                    # Features
                    x = torch.Tensor(feature_matrix[:, t, :])
                    # Targets
                    y = torch.Tensor(feature_matrix[:, t + 1, :])

                    # Determine valid pairs
                    valid_x = x[:, -1] == 1
                    valid_y = y[:, -1] == 1
                    valid_mask = torch.logical_and(valid_x, valid_y)

                    # Save data object to list
                    data = Data(
                        x=x[valid_mask, :-1], y=y[valid_mask, :7], edge_index=None
                    )
                    data["tracks_to_predict"] = torch.where(
                        torch.Tensor(parsed["state/tracks_to_predict"].numpy())[
                            valid_mask
                        ]
                        > 0,
                        True,
                        False,
                    )
                    data["type"] = torch.Tensor(parsed["state/type"].numpy())[
                        valid_mask
                    ].type(torch.LongTensor)

                    data_list.append(data)

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SequentialWaymoTrainDataset(InMemoryDataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        super(SequentialWaymoTrainDataset, self).__init__(
            root, transform, pre_transform
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = os.listdir("data/raw/waymo/train")
        names.sort()
        return names

    @property
    def processed_file_names(self):
        return ["waymo_train_data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "data/raw/waymo/train")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "data/processed/waymo/train/sequential")

    def download(self):
        pass

    def process(self):
        data_list = []
        # Node attributes
        key_values = [
            "x",
            "y",
            "z",
            "velocity_x",
            "velocity_y",
            "bbox_yaw",
            "vel_yaw",
            "width",
            "length",
            "height",
            "valid",
        ]

        # Dimensions
        n_nodes = 128
        n_features = len(key_values)
        n_steps = 91

        # Move through raw files
        for i, raw_path in enumerate(self.raw_file_names):
            # Load file
            dataset = tf.data.TFRecordDataset(
                self.raw_dir + "/" + raw_path, compression_type=""
            )
            for seq_idx, data in enumerate(dataset.as_numpy_iterator()):
                # Parse sequence
                parsed = parse_sequence(data)
                # Allocate tensor for full value array
                feature_matrix = torch.zeros((n_nodes, n_steps, n_features))

                # Fill in all values
                for j, key in enumerate(key_values):
                    # Encode history
                    feature_matrix[:, :10, j] = torch.Tensor(
                        parsed["state/past/" + key].numpy()
                    )
                    feature_matrix[:, 10, j] = torch.Tensor(
                        parsed["state/current/" + key].numpy()
                    ).squeeze()
                    feature_matrix[:, 11:, j] = torch.Tensor(
                        parsed["state/future/" + key].numpy()
                    )

                type = torch.Tensor(parsed["state/type"].numpy()).type(torch.LongTensor)
                mask = torch.where(type > -0.5, True, False)

                data = Data(x=feature_matrix[mask], edge_index=None)
                data["tracks_to_predict"] = torch.where(
                    torch.Tensor(parsed["state/tracks_to_predict"].numpy())[mask] > 0,
                    True,
                    False,
                )
                data["type"] = type[mask]

                data_list.append(data)

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SequentialWaymoValDataset(InMemoryDataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        super(SequentialWaymoValDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = os.listdir("data/raw/waymo/val")
        names.sort()
        return names

    @property
    def processed_file_names(self):
        return ["waymo_val_data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "data/raw/waymo/val")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "data/processed/waymo/val")

    def download(self):
        pass

    def process(self):
        data_list = []
        # Node attributes
        key_values = [
            "x",
            "y",
            "z",
            "velocity_x",
            "velocity_y",
            "bbox_yaw",
            "vel_yaw",
            "width",
            "length",
            "height",
            "valid",
        ]

        # Dimensions
        n_nodes = 128
        n_features = len(key_values)
        n_steps = 91

        # Move through raw files
        for i, raw_path in enumerate(self.raw_file_names):
            # Load file
            dataset = tf.data.TFRecordDataset(
                self.raw_dir + "/" + raw_path, compression_type=""
            )
            for seq_idx, data in enumerate(dataset.as_numpy_iterator()):
                # Parse sequence
                parsed = parse_sequence(data)
                # Allocate tensor for full value array
                feature_matrix = torch.zeros((n_nodes, n_steps, n_features))

                # Fill in all values
                for j, key in enumerate(key_values):
                    # Encode history
                    feature_matrix[:, :10, j] = torch.Tensor(
                        parsed["state/past/" + key].numpy()
                    )
                    feature_matrix[:, 10, j] = torch.Tensor(
                        parsed["state/current/" + key].numpy()
                    ).squeeze()
                    feature_matrix[:, 11:, j] = torch.Tensor(
                        parsed["state/future/" + key].numpy()
                    )

                # Extract agent type
                type = torch.Tensor(parsed["state/type"].numpy()).type(torch.LongTensor)
                # Mask of valid agents for full sequence
                mask = torch.where(type > -0.5, True, False)
                # Create data object
                data = Data(x=feature_matrix[mask], edge_index=None)
                data["tracks_to_predict"] = torch.where(
                    torch.Tensor(parsed["state/tracks_to_predict"].numpy())[mask] > 0,
                    True,
                    False,
                )
                data["type"] = type[mask]
                data_list.append(data)

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class OneStepWaymoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="data/processed/waymo",
        batch_size=1,
        shuffle=False,
        val_batch_size: int = 1,
    ):
        super().__init__(data_dir, batch_size, shuffle)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.val_batch_size = val_batch_size

    @property
    def num_features(self) -> int:
        return 15

    def prepare_data(self):
        OneStepWaymoTrainDataset()
        SequentialWaymoValDataset()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = OneStepWaymoTrainDataset()
        self.val_dataset = SequentialWaymoValDataset()
        # self.test_dataset = SequentialTestDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=16
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self):
        raise NotImplementedError
        # return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def predict_dataloader(self):
        raise NotImplementedError


class SequentialWaymoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="data/processed/waymo",
        batch_size: int = 1,
        shuffle: bool = False,
        val_batch_size: int = 1,
    ):
        super().__init__(data_dir, batch_size, shuffle)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.val_batch_size = val_batch_size

    @property
    def num_features(self) -> int:
        return 11

    def prepare_data(self):
        SequentialWaymoTrainDataset()
        SequentialWaymoValDataset()
        # SequentialTestDataset()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SequentialWaymoTrainDataset()
        self.val_dataset = SequentialWaymoValDataset()
        # self.test_dataset = SequentialTestDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False
        )

    def test_dataloader(self):
        raise NotImplementedError
        # return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def predict_dataloader(self):
        raise NotImplementedError
