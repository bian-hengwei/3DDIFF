import hydra
import lightning as L
import os

from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from x3.dataset.shapenet import ShapeNetDataModule
from x3.utils.lightning_utils import EpochTimeMonitor
from x3.ae.lightning import AELightningModule


def train(config):
    # build shapenet dataset
    datamodule = ShapeNetDataModule(**config.dataset, triplane=True, debug_length=config.trainer.debug_length)
    datamodule.setup()

    # initialize training module
    model = AELightningModule(datamodule.train_dataset.input_dim, **config.ae.params)

    logger = WandbLogger(name=config.exp, save_dir='wandb_log', project='X3', log_model=False, offline=config.trainer.debug)

    callbacks = list()
    if not config.trainer.debug:
        callbacks.append(ModelCheckpoint(dirpath='ckpts', filename=config.exp, monitor='loss', auto_insert_metric_name=False))
        callbacks.append(RichProgressBar())
    callbacks.append(EpochTimeMonitor())

    L.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
    ).fit(model, datamodule=datamodule)


@hydra.main(config_path='conf', config_name='train_triplane', version_base=None)
def main(config: DictConfig):
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print(OmegaConf.to_yaml(config))
    train(config)


if __name__ == '__main__':
    main()
