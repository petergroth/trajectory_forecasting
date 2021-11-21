from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule

if __name__ == "__main__":

    dm = SequentialWaymoDataModule(batch_size=32, shuffle=True)
    dm.setup()
    dataset = dm.train_dataset
    dataset.process()
