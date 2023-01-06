import pytorch_lightning as pl
from cvops.model import CellModel
from cvops.data import CellDataModule


def train_model():
    model = CellModel(init_features=1)
    dm = CellDataModule(batch_size=1, max_images=5, num_workers=0)
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1)
    trainer.fit(model, dm)


if __name__ == "__main__":
    train_model()
