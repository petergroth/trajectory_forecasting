from src.data.dataset_waymo import OneStepWaymoDataModule, SequentialWaymoDataModule

if __name__ == "__main__":

    dm = SequentialWaymoDataModule(batch_size=1, shuffle=True)
    dm.prepare_data()
    dm.setup("test")
    loader = dm.test_dataloader()
    for i in range(2):
        batch = next(iter(loader))
        print(batch.loc[0, 0].item())
        print(batch.loc[0, -1].item())
        print(batch.std[0, 0].item())
        print(batch.std[0, -1].item())
