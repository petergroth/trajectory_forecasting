from data.dataset import OneStepWaymoDataModule, SequentialWaymoDataModule

if __name__ == "__main__":
    print("Processing one-step module")
    dm = OneStepWaymoDataModule()
    dm.prepare_data()

    print("Processing sequential module")
    dm = SequentialWaymoDataModule()
    dm.prepare_data()
