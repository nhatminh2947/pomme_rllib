from ray.rllib.policy import TorchPolicy
from ray.rllib.utils.annotations import override


class PBTParamsMixin:
    def __init__(self, config):
        self.cur_lr_val = config['lr']
        self.cur_lr = 0.0001

    @override(TorchPolicy)
    def optimizer(self):
        for opt in self._optimizers:
            for p in opt.param_groups:
                p["lr"] = self.cur_lr
        return self._optimizers
