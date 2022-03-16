import pytorch_lightning as pl
import os

import torch


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self,
            save_step_frequency,
            prefix="N-Step-Checkpoint",
            use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        log_dir = trainer.logger.log_dir
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{log_dir}/checkpoints/{self.prefix}_epoch={epoch}_step={global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1,3,1,1)
        self.std = torch.tensor(std).view(1,3,1,1)

    def __call__(self, tensor):
        tensor_out = tensor*self.std.type_as(tensor)+self.mean.type_as(tensor)
        return tensor_out


def load_module_params_from_ckpt(m_module, module_name, ckpt_path, isStrict=True):
    state_dict = torch.load(ckpt_path)['state_dict']
    name_len = len(module_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key[:name_len] == module_name:
            new_state_dict[key[name_len + 1:]] = value
    m_module.load_state_dict(new_state_dict, isStrict)
    return m_module


def fixParameter(m_module):
    m_module.eval()
    for param in m_module.parameters():
        param.requires_grad = False
    return m_module
