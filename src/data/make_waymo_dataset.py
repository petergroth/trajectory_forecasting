from src.data.dataset import OneStepWaymoDataModule, SequentialWaymoDataModule

if __name__ == "__main__":

    dm = OneStepWaymoDataModule()
    dm.setup()

    # print("Processing sequential module")
    # dm = SequentialWaymoDataModule()
    # dm.prepare_data()


    loader = dm.train_dataloader()
    batch = next(iter(loader))
