from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from cvops.model import CellModel
from cvops.data import CellDataModule


def train_model(save_dir=Path("./data/logs")):
    model = CellModel(init_features=1)
    data_module = CellDataModule(batch_size=1, max_images=5, num_workers=0)

    # Ugly hack because of Lightning logger that essentially join save_dir and name
    ll_save_dir = save_dir.parent
    ll_name = str(save_dir.name)
    tensorboardlogger = TensorBoardLogger(save_dir=ll_save_dir, name=ll_name)
    csvlogger = CSVLogger(save_dir=ll_save_dir, name=ll_name)

    loggers = [tensorboardlogger, csvlogger]
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, default_root_dir=save_dir, logger=loggers)
    trainer.fit(model=model, datamodule=data_module)

    df = pd.read_csv(csvlogger.experiment.metrics_file_path)
    df.plot(
        x="step", y="valid_step_loss", kind="scatter", title="Evolution of valid loss per training step"
    ).get_figure().savefig(Path(csvlogger.log_dir) / "validation_loss_per_step")


if __name__ == "__main__":
    train_model()
