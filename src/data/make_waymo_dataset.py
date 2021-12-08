from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule

if __name__ == "__main__":

    dm = SequentialWaymoDataModule(batch_size=32, shuffle=True)
    dm.setup()

    loader = dm.train_dataloader()

    batch = next(iter(loader))