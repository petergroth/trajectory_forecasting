from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule

if __name__ == "__main__":

    # dm = OneStepWaymoDataModule()
    # dm.setup()
    #

    dm = SequentialWaymoDataModule(batch_size=2)
    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))
