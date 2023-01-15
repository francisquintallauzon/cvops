from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from dvclive.lightning import DVCLiveLogger
from cvops.model import CellModel
from cvops.data.data import BBC5v1DataModule


def train_model(save_dir=Path("./data/logs"), limit_train_batches=0.1, limit_valid_batches=0.1):
    model = CellModel(init_features=1)
    data_module = BBC5v1DataModule(batch_size=10, num_workers=0)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Ugly hack because of Lightning logger that essentially join save_dir and name
    ll_save_dir = save_dir.parent
    ll_name = str(save_dir.name)
    tensorboard_logger = TensorBoardLogger(save_dir=ll_save_dir, name=ll_name)
    csv_logger = CSVLogger(save_dir=ll_save_dir, name=ll_name)
    dvc_logger = DVCLiveLogger(dir=save_dir)

    loggers = [tensorboard_logger, csv_logger, dvc_logger]
    trainer = pl.Trainer(
        max_epochs=6,
        log_every_n_steps=10,
        default_root_dir=save_dir,
        logger=loggers,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_valid_batches,
    )
    trainer.fit(model=model, datamodule=data_module)

    # Create a plot of Validation Loss per training step png file
    df = pd.read_csv(csv_logger.experiment.metrics_file_path)
    df.plot(
        x="step", y="valid_step_loss", kind="scatter", title="Evolution of valid loss per training step"
    ).get_figure().savefig(Path(csv_logger.log_dir) / "validation_loss_per_step")


if __name__ == "__main__":
    train_model()
