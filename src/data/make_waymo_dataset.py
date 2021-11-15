from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule

if __name__ == "__main__":

    dm = OneStepWaymoDataModule(batch_size=128, shuffle=True)
    dm.setup()

    #
    # dm = SequentialWaymoDataModule(batch_size=2)
    # dm.setup()

    loader = dm.val_dataloader()
    batch = next(iter(loader))
    # for i in enumerate(loader):
    #     print(batch.loc[:2])

    # dataset = dm.val_dataset
    # dataset.process()
