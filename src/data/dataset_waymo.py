import math
import os
import os.path as osp
from typing import Optional

import numpy as np
import pytorch_lightning as pl
# import tensorflow as tf
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader

# from src.utils import parse_sequence


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

                # Process yaw-values into [-pi, pi]
                x_yaws = feature_matrix[:, :, 5:7]
                x_yaws[x_yaws > 0] = (
                    torch.fmod(x_yaws[x_yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                    - math.pi
                )
                x_yaws[x_yaws < 0] = (
                    torch.fmod(x_yaws[x_yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                    + math.pi
                )
                feature_matrix[:, :, 5:7] = x_yaws

                # Values and mask to use for normalisation
                vals = feature_matrix[:, :11, :-1]
                mask = feature_matrix[:, :11, -1].bool()
                loc = torch.mean(vals[mask], dim=0)
                std = torch.std(vals[mask], dim=0)

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

                    # Combine features with types
                    node_features = torch.cat(
                        [
                            x[valid_mask, :-1],
                            one_hot(
                                torch.Tensor(parsed["state/type"].numpy())[
                                    valid_mask
                                ].type(torch.LongTensor),
                                num_classes=5,
                            ),
                        ],
                        dim=1,
                    )

                    # Save data object to list
                    data = Data(x=node_features, y=y[valid_mask, :7], edge_index=None)
                    data["loc"] = loc.unsqueeze(0)
                    data["std"] = std.unsqueeze(0)

                    data_list.append(data)

        # Collate and save
        data, slices = self.collate(data_list)
        # GLOBAL SCALER
        # global_scale = data.std[:, [0, 1, 2, 3, 4, 7, 8, 9]].mean()
        # global_scaler = 8.025897979736328
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
            # "z",
            "velocity_x",
            "velocity_y",
            # "bbox_yaw",
            # "vel_yaw",
            "width",
            "length",
            "height",
            "valid",
        ]

        # Dimensions
        n_nodes = 128
        n_features = len(key_values)
        n_steps = 91
        # total_sum = np.zeros(16)
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
                feature_matrix = np.zeros((n_nodes, n_steps, n_features))

                # Fill in all values
                for j, key in enumerate(key_values):
                    # Encode history
                    feature_matrix[:, :10, j] = parsed["state/past/" + key].numpy()
                    feature_matrix[:, 10, j] = (
                        parsed["state/current/" + key].numpy().squeeze()
                    )
                    feature_matrix[:, 11:, j] = parsed["state/future/" + key].numpy()

                roadgraph = parsed["roadgraph_samples/xyz"].numpy()
                roadgraph_type = parsed["roadgraph_samples/type"].numpy()
                roadgraph_mask = np.array(
                    parsed["roadgraph_samples/valid"].numpy().squeeze(), dtype=bool
                )
                roadgraph = roadgraph[roadgraph_mask][:, :2]
                roadgraph_type = roadgraph_type[roadgraph_mask][:, :2]

                width = 150

                # Compute span of area
                all_x = feature_matrix[:, :11, 0][
                    feature_matrix[:, :11, -1].astype(bool)
                ]
                all_y = feature_matrix[:, :11, 1][
                    feature_matrix[:, :11, -1].astype(bool)
                ]
                center_x, center_y = np.mean(all_x), np.mean(all_y)

                # channels = [1, 2, 3, 6, 7, 12, 15, 16, 17, 18, 19]
                channel_dict = {
                    "LaneCenterFreeway": 1,
                    "LaneCenterSurfaceStreet": 2,
                    "LaneCenterBikeLane": 3,
                    "RoadLineBrokenSingleWhite": 6,
                    "RoadLineSolidSingleWhite": 7,
                    "RoadLineSolidDoubleWhite": 8,
                    "RoadLineBrokenSingleYellow": 9,
                    "RoadLineBrokenDoubleYellow": 10,
                    "RoadlineSolidSingleYellow": 11,
                    "RoadlineSolidDoubleYellow": 12,
                    "RoadLinePassingDoubleYellow": 13,
                    "RoadEdgeBoundary": 15,
                    "RoadEdgeMedian": 16,
                    "StopSign": 17,
                    "Crosswalk": 18,
                    "SpeedBump": 19,
                }
                # channels = np.arange(20)
                u = np.zeros((len(channel_dict), width * 2, width * 2), dtype=np.byte)
                for j, key in enumerate(channel_dict):
                    road_edge_mask = (roadgraph_type == channel_dict[key]).squeeze()
                    roadgraph_i = roadgraph[road_edge_mask]
                    # Histogram of roadgraph
                    u_i, _, _ = np.histogram2d(
                        x=roadgraph_i[:, 0],
                        y=roadgraph_i[:, 1],
                        range=[
                            [center_x - width / 2, center_x + width / 2],
                            [center_y - width / 2, center_y + width / 2],
                        ],
                        bins=[width * 2, width * 2],
                    )
                    # Binarise
                    u_i[u_i > 0] = 1
                    # total_sum[i] += u_i.sum()
                    u_i = u_i.astype(np.byte)
                    u[j] = u_i.T

                # Combine similar road features for smaller memory footprint
                u_center = u[0] + u[1]
                u_bikelane = u[2]
                u_broken_white = u[3]
                u_solid_white = u[4] + u[5]
                u_broken_yellow = u[6] + u[7] + u[10]
                u_solid_yellow = u[8] + u[9]
                u_edges = u[11] + u[12]
                u_obstacles = u[13] + u[14] + u[15]

                u = np.stack(
                    [
                        u_center,
                        u_bikelane,
                        u_broken_white,
                        u_solid_white,
                        u_broken_yellow,
                        u_solid_yellow,
                        u_edges,
                        u_obstacles,
                    ]
                )

                # for i in range(u.shape[0]):
                #     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                #     ax.imshow(u[i], origin='lower', extent=(center_x - width / 2,
                #                                              center_x + width / 2,
                #                                              center_y - width / 2,
                #                                              center_y + width / 2),
                #               interpolation=None, cmap='Greys'
                #              )
                #     ax.scatter(x=all_x, y=all_y, color='#f58a00')
                #     plt.title(i)
                #     plt.show()
                #
                # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                # ax.imshow(np.sum(u,axis=0), origin='lower', extent=(center_x - width / 2,
                #                                         center_x + width / 2,
                #                                         center_y - width / 2,
                #                                         center_y + width / 2),
                #           interpolation=None, cmap='Greys'
                #           )
                # ax.scatter(x=all_x, y=all_y, color='#f58a00')
                # # plt.title(i)
                # plt.show()

                feature_matrix = torch.Tensor(feature_matrix)
                # # Process yaw-values into [-pi, pi]
                # x_yaws = feature_matrix[:, :, 5:7]
                # x_yaws[x_yaws > 0] = (
                #     torch.fmod(x_yaws[x_yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                #     - math.pi
                # )
                # x_yaws[x_yaws < 0] = (
                #     torch.fmod(x_yaws[x_yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                #     + math.pi
                # )
                # feature_matrix[:, :, 5:7] = x_yaws

                # Values and mask to use for normalisation
                vals = feature_matrix[:, :11, :-1]
                mask = feature_matrix[:, :11, -1].bool()
                loc = torch.mean(vals[mask], dim=0)
                std = torch.std(vals[mask], dim=0)

                # Extract agent type
                type = torch.Tensor(parsed["state/type"].numpy()).type(torch.LongTensor)
                # Mask of valid agents for full sequence
                mask = torch.where(type > -0.5, True, False)

                data = Data(x=feature_matrix[mask], edge_index=None)
                data["tracks_to_predict"] = torch.where(
                    torch.Tensor(parsed["state/tracks_to_predict"].numpy())[mask] > 0,
                    True,
                    False,
                )
                # Add agent type as one-hot encoded
                data["type"] = one_hot(type[mask], num_classes=5)
                data["loc"] = loc.unsqueeze(0)
                data["std"] = std.unsqueeze(0)
                data["u"] = torch.ByteTensor(u).unsqueeze(0)

                data_list.append(data)

            # print(f"Finished file {i+1}")

        # Collate and save
        # print("Attempting to collate")
        data, slices = self.collate(data_list)
        del data_list, feature_matrix, parsed, u, roadgraph, dataset
        # print("Attempting to save")
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
            # "z",
            "velocity_x",
            "velocity_y",
            # "bbox_yaw",
            # "vel_yaw",
            "width",
            "length",
            "height",
            "valid",
        ]

        # Dimensions
        n_nodes = 128
        n_features = len(key_values)
        n_steps = 91
        # total_sum = np.zeros(16)
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
                feature_matrix = np.zeros((n_nodes, n_steps, n_features))

                # Fill in all values
                for j, key in enumerate(key_values):
                    # Encode history
                    feature_matrix[:, :10, j] = parsed["state/past/" + key].numpy()
                    feature_matrix[:, 10, j] = (
                        parsed["state/current/" + key].numpy().squeeze()
                    )
                    feature_matrix[:, 11:, j] = parsed["state/future/" + key].numpy()

                roadgraph = parsed["roadgraph_samples/xyz"].numpy()
                roadgraph_type = parsed["roadgraph_samples/type"].numpy()
                roadgraph_mask = np.array(
                    parsed["roadgraph_samples/valid"].numpy().squeeze(), dtype=bool
                )
                roadgraph = roadgraph[roadgraph_mask][:, :2]
                roadgraph_type = roadgraph_type[roadgraph_mask][:, :2]

                width = 150

                # Compute span of area
                all_x = feature_matrix[:, :11, 0][
                    feature_matrix[:, :11, -1].astype(bool)
                ]
                all_y = feature_matrix[:, :11, 1][
                    feature_matrix[:, :11, -1].astype(bool)
                ]
                center_x, center_y = np.mean(all_x), np.mean(all_y)

                # channels = [1, 2, 3, 6, 7, 12, 15, 16, 17, 18, 19]
                channel_dict = {
                    "LaneCenterFreeway": 1,
                    "LaneCenterSurfaceStreet": 2,
                    "LaneCenterBikeLane": 3,
                    "RoadLineBrokenSingleWhite": 6,
                    "RoadLineSolidSingleWhite": 7,
                    "RoadLineSolidDoubleWhite": 8,
                    "RoadLineBrokenSingleYellow": 9,
                    "RoadLineBrokenDoubleYellow": 10,
                    "RoadlineSolidSingleYellow": 11,
                    "RoadlineSolidDoubleYellow": 12,
                    "RoadLinePassingDoubleYellow": 13,
                    "RoadEdgeBoundary": 15,
                    "RoadEdgeMedian": 16,
                    "StopSign": 17,
                    "Crosswalk": 18,
                    "SpeedBump": 19,
                }
                # channels = np.arange(20)
                u = np.zeros((len(channel_dict), width * 2, width * 2), dtype=np.byte)
                for j, key in enumerate(channel_dict):
                    road_edge_mask = (roadgraph_type == channel_dict[key]).squeeze()
                    roadgraph_i = roadgraph[road_edge_mask]
                    # Histogram of roadgraph
                    u_i, _, _ = np.histogram2d(
                        x=roadgraph_i[:, 0],
                        y=roadgraph_i[:, 1],
                        range=[
                            [center_x - width / 2, center_x + width / 2],
                            [center_y - width / 2, center_y + width / 2],
                        ],
                        bins=[width * 2, width * 2],
                    )
                    # Binarise
                    u_i[u_i > 0] = 1
                    # total_sum[i] += u_i.sum()
                    u_i = u_i.astype(np.byte)
                    u[j] = u_i.T

                # Combine similar road features for smaller memory footprint
                u_center = u[0] + u[1]
                u_bikelane = u[2]
                u_broken_white = u[3]
                u_solid_white = u[4] + u[5]
                u_broken_yellow = u[6] + u[7] + u[10]
                u_solid_yellow = u[8] + u[9]
                u_edges = u[11] + u[12]
                u_obstacles = u[13] + u[14] + u[15]

                u = np.stack(
                    [
                        u_center,
                        u_bikelane,
                        u_broken_white,
                        u_solid_white,
                        u_broken_yellow,
                        u_solid_yellow,
                        u_edges,
                        u_obstacles,
                    ]
                )

                # for i in range(u.shape[0]):
                #     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                #     ax.imshow(u[i], origin='lower', extent=(center_x - width / 2,
                #                                              center_x + width / 2,
                #                                              center_y - width / 2,
                #                                              center_y + width / 2),
                #               interpolation=None, cmap='Greys'
                #              )
                #     ax.scatter(x=all_x, y=all_y, color='#f58a00')
                #     plt.title(i)
                #     plt.show()
                #
                # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                # ax.imshow(np.sum(u,axis=0), origin='lower', extent=(center_x - width / 2,
                #                                         center_x + width / 2,
                #                                         center_y - width / 2,
                #                                         center_y + width / 2),
                #           interpolation=None, cmap='Greys'
                #           )
                # ax.scatter(x=all_x, y=all_y, color='#f58a00')
                # # plt.title(i)
                # plt.show()

                feature_matrix = torch.Tensor(feature_matrix)
                # # Process yaw-values into [-pi, pi]
                # x_yaws = feature_matrix[:, :, 5:7]
                # x_yaws[x_yaws > 0] = (
                #     torch.fmod(x_yaws[x_yaws > 0] + math.pi, torch.tensor(2 * math.pi))
                #     - math.pi
                # )
                # x_yaws[x_yaws < 0] = (
                #     torch.fmod(x_yaws[x_yaws < 0] - math.pi, torch.tensor(2 * math.pi))
                #     + math.pi
                # )
                # feature_matrix[:, :, 5:7] = x_yaws

                # Values and mask to use for normalisation
                vals = feature_matrix[:, :11, :-1]
                mask = feature_matrix[:, :11, -1].bool()
                loc = torch.mean(vals[mask], dim=0)
                std = torch.std(vals[mask], dim=0)

                # Extract agent type
                type = torch.Tensor(parsed["state/type"].numpy()).type(torch.LongTensor)
                # Mask of valid agents for full sequence
                mask = torch.where(type > -0.5, True, False)

                data = Data(x=feature_matrix[mask], edge_index=None)
                data["tracks_to_predict"] = torch.where(
                    torch.Tensor(parsed["state/tracks_to_predict"].numpy())[mask] > 0,
                    True,
                    False,
                )
                # Add agent type as one-hot encoded
                data["type"] = one_hot(type[mask], num_classes=5)
                data["loc"] = loc.unsqueeze(0)
                data["std"] = std.unsqueeze(0)
                data["u"] = torch.ByteTensor(u).unsqueeze(0)

                data_list.append(data)

            # print(f"Finished file {i+1}")

        # Collate and save
        # print("Attempting to collate")
        data, slices = self.collate(data_list)
        del data_list, feature_matrix, parsed, u, roadgraph, dataset
        # print("Attempting to save")
        torch.save((data, slices), self.processed_paths[0])


class SequentialWaymoTestDataset(InMemoryDataset):
    def __init__(self, root="", transform=None, pre_transform=None):
        super(SequentialWaymoTestDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = os.listdir("data/raw/waymo/test")
        names.sort()
        return names

    @property
    def processed_file_names(self):
        return ["waymo_test_data.pt"]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "data/raw/waymo/test")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "data/processed/waymo/test")

    def download(self):
        pass

    def process(self):
        data_list = []
        # Node attributes
        key_values = [
            "x",
            "y",
            # "z",
            "velocity_x",
            "velocity_y",
            # "bbox_yaw",
            # "vel_yaw",
            "width",
            "length",
            "height",
            "valid",
        ]

        # Dimensions
        n_nodes = 128
        n_features = len(key_values)
        n_steps = 91
        # total_sum = np.zeros(16)
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
                feature_matrix = np.zeros((n_nodes, n_steps, n_features))

                # Fill in all values
                for j, key in enumerate(key_values):
                    # Encode history
                    feature_matrix[:, :10, j] = parsed["state/past/" + key].numpy()
                    feature_matrix[:, 10, j] = (
                        parsed["state/current/" + key].numpy().squeeze()
                    )
                    feature_matrix[:, 11:, j] = parsed["state/future/" + key].numpy()

                roadgraph = parsed["roadgraph_samples/xyz"].numpy()
                roadgraph_type = parsed["roadgraph_samples/type"].numpy()
                roadgraph_mask = np.array(
                    parsed["roadgraph_samples/valid"].numpy().squeeze(), dtype=bool
                )
                roadgraph = roadgraph[roadgraph_mask][:, :2]
                roadgraph_type = roadgraph_type[roadgraph_mask][:, :2]

                width = 150

                # Compute span of area
                all_x = feature_matrix[:, :11, 0][
                    feature_matrix[:, :11, -1].astype(bool)
                ]
                all_y = feature_matrix[:, :11, 1][
                    feature_matrix[:, :11, -1].astype(bool)
                ]
                center_x, center_y = np.mean(all_x), np.mean(all_y)

                channel_dict = {
                    "LaneCenterFreeway": 1,
                    "LaneCenterSurfaceStreet": 2,
                    "LaneCenterBikeLane": 3,
                    "RoadLineBrokenSingleWhite": 6,
                    "RoadLineSolidSingleWhite": 7,
                    "RoadLineSolidDoubleWhite": 8,
                    "RoadLineBrokenSingleYellow": 9,
                    "RoadLineBrokenDoubleYellow": 10,
                    "RoadlineSolidSingleYellow": 11,
                    "RoadlineSolidDoubleYellow": 12,
                    "RoadLinePassingDoubleYellow": 13,
                    "RoadEdgeBoundary": 15,
                    "RoadEdgeMedian": 16,
                    "StopSign": 17,
                    "Crosswalk": 18,
                    "SpeedBump": 19,
                }

                u = np.zeros((len(channel_dict), width * 2, width * 2), dtype=np.byte)
                for j, key in enumerate(channel_dict):
                    road_edge_mask = (roadgraph_type == channel_dict[key]).squeeze()
                    roadgraph_i = roadgraph[road_edge_mask]
                    # Histogram of roadgraph
                    u_i, _, _ = np.histogram2d(
                        x=roadgraph_i[:, 0],
                        y=roadgraph_i[:, 1],
                        range=[
                            [center_x - width / 2, center_x + width / 2],
                            [center_y - width / 2, center_y + width / 2],
                        ],
                        bins=[width * 2, width * 2],
                    )
                    # Binarise
                    u_i[u_i > 0] = 1
                    u_i = u_i.astype(np.byte)
                    u[j] = u_i.T

                # Combine similar road features for smaller memory footprint
                u_center = u[0] + u[1]
                u_bikelane = u[2]
                u_broken_white = u[3]
                u_solid_white = u[4] + u[5]
                u_broken_yellow = u[6] + u[7] + u[10]
                u_solid_yellow = u[8] + u[9]
                u_edges = u[11] + u[12]
                u_obstacles = u[13] + u[14] + u[15]

                u = np.stack(
                    [
                        u_center,
                        u_bikelane,
                        u_broken_white,
                        u_solid_white,
                        u_broken_yellow,
                        u_solid_yellow,
                        u_edges,
                        u_obstacles,
                    ]
                )

                feature_matrix = torch.Tensor(feature_matrix)

                # Values and mask to use for normalisation
                vals = feature_matrix[:, :11, :-1]
                mask = feature_matrix[:, :11, -1].bool()
                loc = torch.mean(vals[mask], dim=0)
                std = torch.std(vals[mask], dim=0)

                # Extract agent type
                type = torch.Tensor(parsed["state/type"].numpy()).type(torch.LongTensor)
                # Mask of valid agents for full sequence
                mask = torch.where(type > -0.5, True, False)

                data = Data(x=feature_matrix[mask], edge_index=None)
                data["tracks_to_predict"] = torch.where(
                    torch.Tensor(parsed["state/tracks_to_predict"].numpy())[mask] > 0,
                    True,
                    False,
                )
                # Add agent type as one-hot encoded
                data["type"] = one_hot(type[mask], num_classes=5)
                data["loc"] = loc.unsqueeze(0)
                data["std"] = std.unsqueeze(0)
                data["u"] = torch.ByteTensor(u).unsqueeze(0)

                data_list.append(data)

        # Collate and save
        data, slices = self.collate(data_list)
        del data_list, feature_matrix, parsed, u, roadgraph, dataset
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

    def prepare_data(self):
        OneStepWaymoTrainDataset()
        SequentialWaymoValDataset()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = OneStepWaymoTrainDataset()
        self.val_dataset = SequentialWaymoValDataset()
        self.test_dataset = SequentialWaymoTestDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=4,
        )

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

    def prepare_data(self):
        SequentialWaymoTrainDataset()
        SequentialWaymoValDataset()
        SequentialWaymoTestDataset()

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = SequentialWaymoTrainDataset()
            self.val_dataset = SequentialWaymoValDataset()
        elif stage == "validate":
            self.val_dataset = SequentialWaymoValDataset()
        elif stage == "test":
            self.test_dataset = SequentialWaymoTestDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=6,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=6,
        )
