from data.dataset_nbody import OneStepNBodyDataModule, SequentialNBodyDataModule


def main():
    batch_size = 64
    dm = OneStepNBodyDataModule(batch_size=batch_size)
    dm.prepare_data()
    dm.setup()

    dm = SequentialNBodyDataModule(batch_size=batch_size)
    dm.prepare_data()
    dm.setup()


if __name__ == "__main__":
    main()
