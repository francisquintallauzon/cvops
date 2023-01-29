from cvops.data.data import BBC5v1DataModule
from pathlib import Path


def test_BBC5v1DataModule():
    # TODO: Add some relevant validation criteria
    bbc5v1dm = BBC5v1DataModule()
    bbc5v1dm.prepare_data()
    bbc5v1dm.setup(stage="fit")

    list_of_filename_list = [bbc5v1dm.filenames_train, bbc5v1dm.filenames_valid, bbc5v1dm.filenames_test]

    for fn_lst in list_of_filename_list:
        assert len(fn_lst) > 1
        for fn in fn_lst:
            assert Path(fn).suffix.casefold() == ".tif"

    print("I am super happy with the test")
