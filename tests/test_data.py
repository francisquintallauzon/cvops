from cvops.data import CellDataModule


def test_CellDataModule():
    # TODO: Add some relevant validation criteria
    celldm = CellDataModule()
    celldm.prepare_data()
    celldm.setup(stage="fit")

    print("I am super happy with the test")
