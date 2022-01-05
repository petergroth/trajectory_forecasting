import os
import os.path as osp
from typing import Optional

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader

from src.utils import generate_fully_connected_edges, load_simulations


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
        assert len(self.raw_file_names) == 2000
        data_list = []

        # Move through training samples and save all graphs separately
        for i, raw_path in enumerate(self.raw_file_names[:1000]):
            full_arr = load_simulations(
                self.raw_dir + "/" + raw_path
            )  # [n_steps, n_nodes, 5]
            n_steps = full_arr.shape[0]

            for t in range(n_steps - 1):
                # Features
                x = torch.Tensor(full_arr[t, :, :])
                # Targets
                y = torch.Tensor(full_arr[t + 1, :, :])
                data = Data(x=x, y=y, edge_index=None)
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
        return osp.join(self.root, "data/processed/nbody/sequential")

    def download(self):
        pass

    def process(self):
        assert len(self.raw_file_names) == 1500
        data_list = []

        for j, raw_path in enumerate(self.raw_file_names[:1000]):
            full_arr = load_simulations(
                self.raw_dir + "/" + raw_path
            )  # [n_steps, n_nodes, 5]
            n_steps = full_arr.shape[0]
            n_nodes = full_arr.shape[1]

            # Use all x values
            x = torch.Tensor(full_arr).permute(1, 0, 2)
            data = Data(x=x, edge_index=None, edge_attr=None)
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
        return osp.join(self.root, "data/processed/nbody/sequential")

    def download(self):
        pass

    def process(self):
        assert len(self.raw_file_names) == 2000
        data_list = []

        for j, raw_path in enumerate(self.raw_file_names[1000:1500]):
            full_arr = load_simulations(
                self.raw_dir + "/" + raw_path
            )  # [n_steps, n_nodes, 5]

            # Use all x values
            x = torch.Tensor(full_arr).permute(1, 0, 2)
            data = Data(x=x, edge_index=None)
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
        return osp.join(self.root, "data/processed/nbody/sequential")

    def download(self):
        pass

    def process(self):
        assert len(self.raw_file_names) == 2000
        data_list = []

        for j, raw_path in enumerate(self.raw_file_names[1500:]):
            full_arr = load_simulations(self.raw_dir + "/" + raw_path)
            # Use all x values
            x = torch.Tensor(full_arr).permute(1, 0, 2)
            data = Data(x=x, edge_index=None)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class OneStepNBodyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/processed/nbody",
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
        OneStepTrainDataset()
        SequentialValDataset()
        SequentialTestDataset()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = OneStepTrainDataset()
        self.val_dataset = SequentialValDataset()
        self.test_dataset = SequentialTestDataset()
        assert len(self.val_dataset) == 500
        assert len(self.val_dataset) == 500
        assert len(self.train_dataset) == 1000 * 90

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.val_batch_size, shuffle=False
        )


class SequentialNBodyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/processed/nbody",
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
        assert len(self.val_dataset) == 500
        assert len(self.test_dataset) == 500
        assert len(self.train_dataset) == 1000

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
        return DataLoader(
            self.test_dataset, batch_size=self.val_batch_size, shuffle=False
        )
