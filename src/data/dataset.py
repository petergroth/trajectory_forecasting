import os
import os.path as osp
import torch
from utils import load_simulations, parse_sequence, generate_fully_connected_edges
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import tensorflow as tf
from typing import Optional


class OneStepTrainDataset(InMemoryDataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        super(OneStepTrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = os.listdir("data/raw/nbody")
        names.sort()
        return names

    @property
    def processed_file_names(self):
        return ["nbody_train_data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "data/raw/nbody")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "data/processed/nbody/onestep")

    def download(self):
        pass

    def process(self):
        assert len(self.raw_file_names) == 10000
        data_list = []

        # Move through training samples and save all graphs separately
        for i, raw_path in enumerate(self.raw_file_names[:5000]):
            full_arr = load_simulations(
                self.raw_dir + "/" + raw_path
            )  # [n_steps, n_nodes, 5]
            n_steps = full_arr.shape[0]
            n_nodes = full_arr.shape[1]

            # Common adjacency matrix for all time steps
            edge_index = torch.LongTensor(generate_fully_connected_edges(n_nodes))

            for t in range(n_steps - 1):
                # Features
                x = torch.Tensor(full_arr[t, :, :])
                # Targets
                y = torch.Tensor(full_arr[t + 1, :, :])
                data = Data(x=x, y=y, edge_index=edge_index)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SequentialTrainDataset(InMemoryDataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        super(SequentialTrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = os.listdir("data/raw/nbody")
        names.sort()
        return names

    @property
    def processed_file_names(self):
        return ["nbody_train_data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "data/raw/nbody")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "data/processed/nbody/temporal")

    def download(self):
        pass

    def process(self):
        assert len(self.raw_file_names) == 10000
        data_list = []

        for j, raw_path in enumerate(self.raw_file_names[:5000]):
            full_arr = load_simulations(
                self.raw_dir + "/" + raw_path
            )  # [n_steps, n_nodes, 5]
            n_steps = full_arr.shape[0]
            n_nodes = full_arr.shape[1]

            # Common adjacency matrix for all time steps
            edge_index = torch.LongTensor(generate_fully_connected_edges(n_nodes))
            # Use all x values
            x = torch.Tensor(full_arr).permute(1, 0, 2)
            data = Data(x=x, edge_index=edge_index, edge_attr=None)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SequentialValDataset(InMemoryDataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        super(SequentialValDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = os.listdir("data/raw/nbody")
        names.sort()
        return names

    @property
    def processed_file_names(self):
        return ["nbody_val_data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "data/raw/nbody")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "data/processed/nbody/temporal")

    def download(self):
        pass

    def process(self):
        assert len(self.raw_file_names) == 10000
        data_list = []

        for j, raw_path in enumerate(self.raw_file_names[5000:7500]):
            full_arr = load_simulations(
                self.raw_dir + "/" + raw_path
            )  # [n_steps, n_nodes, 5]
            n_steps = full_arr.shape[0]
            n_nodes = full_arr.shape[1]

            # Common adjacency matrix for all time steps
            edge_index = torch.LongTensor(generate_fully_connected_edges(n_nodes))
            # Use all x values
            x = torch.Tensor(full_arr).permute(1, 0, 2)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SequentialTestDataset(InMemoryDataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        super(SequentialTestDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = os.listdir("data/raw/nbody")
        names.sort()
        return names

    @property
    def processed_file_names(self):
        return ["nbody_test_data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "data/raw/nbody")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "data/processed/nbody/temporal")

    def download(self):
        pass

    def process(self):
        assert len(self.raw_file_names) == 10000
        data_list = []

        for j, raw_path in enumerate(self.raw_file_names[7500:]):
            full_arr = load_simulations(
                self.raw_dir + "/" + raw_path
            )  # [n_steps, n_nodes, 5]
            n_steps = full_arr.shape[0]
            n_nodes = full_arr.shape[1]

            # Common adjacency matrix for all time steps
            edge_index = torch.LongTensor(generate_fully_connected_edges(n_nodes))
            # Use all x values
            x = torch.Tensor(full_arr).permute(1, 0, 2)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class OneStepNBodyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="data/processed/nbody",
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
        return 5

    def prepare_data(self):
        OneStepTrainDataset()
        SequentialValDataset()
        SequentialTestDataset()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = OneStepTrainDataset()
        self.val_dataset = SequentialValDataset()
        self.test_dataset = SequentialTestDataset()
        assert len(self.val_dataset) == len(self.test_dataset) == 2500
        assert len(self.train_dataset) == 450000

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def predict_dataloader(self):
        raise NotImplementedError


class SequentialNBodyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="data/processed/nbody",
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
        return 5

    def prepare_data(self):
        SequentialTrainDataset()
        SequentialValDataset()
        SequentialTestDataset()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SequentialTrainDataset()
        self.val_dataset = SequentialValDataset()
        self.test_dataset = SequentialTestDataset()
        assert len(self.val_dataset) == len(self.test_dataset) == 2500
        assert len(self.train_dataset) == 5000

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
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def predict_dataloader(self):
        raise NotImplementedError


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
        data_list = []
        # Node attributes
        key_values = [
            "x",
            "y",
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

        edge_index = None

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
                    feature_matrix[:, :10, i] = torch.Tensor(
                        parsed["state/past/" + key].numpy()
                    )
                    feature_matrix[:, 10, i] = torch.Tensor(
                        parsed["state/current/" + key].numpy()
                    ).squeeze()
                    feature_matrix[:, 11:, i] = torch.Tensor(
                        parsed["state/future/" + key].numpy()
                    )

                # Determine input/target pairs
                for t in range(n_steps - 1):
                    # Features
                    x = torch.Tensor(feature_matrix[:, t, :])
                    # Targets
                    y = torch.Tensor(feature_matrix[:, t + 1, :])
                    data = Data(x=x, y=y, edge_index=edge_index)
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
                    feature_matrix[:, :10, i] = torch.Tensor(
                        parsed["state/past/" + key].numpy()
                    )
                    feature_matrix[:, 10, i] = torch.Tensor(
                        parsed["state/current/" + key].numpy()
                    ).squeeze()
                    feature_matrix[:, 11:, i] = torch.Tensor(
                        parsed["state/future/" + key].numpy()
                    )

                data = Data(x=feature_matrix, edge_index=None)
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
                    feature_matrix[:, :10, i] = torch.Tensor(
                        parsed["state/past/" + key].numpy()
                    )
                    feature_matrix[:, 10, i] = torch.Tensor(
                        parsed["state/current/" + key].numpy()
                    ).squeeze()
                    feature_matrix[:, 11:, i] = torch.Tensor(
                        parsed["state/future/" + key].numpy()
                    )

                data = Data(x=feature_matrix, edge_index=None)
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
        return 10

    def prepare_data(self):
        OneStepWaymoTrainDataset()
        SequentialWaymoValDataset()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = OneStepWaymoTrainDataset()
        self.val_dataset = SequentialWaymoValDataset()
        # self.test_dataset = SequentialTestDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
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
        return 5

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
