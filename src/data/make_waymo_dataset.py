from src.data.dataset_waymo import (OneStepWaymoDataModule,
                                    SequentialWaymoDataModule)

if __name__ == "__main__":

    dm = SequentialWaymoDataModule(batch_size=1, shuffle=True)
    dm.setup()


    # dm.train_dataset.process()
    # loader = dm.val_dataloader()
    #
    # batch = next(iter(loader))
