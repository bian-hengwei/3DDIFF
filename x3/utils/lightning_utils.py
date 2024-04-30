import time

from lightning.pytorch.callbacks import Callback

class EpochTimeMonitor(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        end_time = time.time()
        epoch_time = end_time - self.start_time
        trainer.logger.log_metrics({'epoch_duration': epoch_time})
