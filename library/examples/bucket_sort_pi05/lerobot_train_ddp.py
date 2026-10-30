import functools

import accelerate.utils

accelerate.utils.DistributedDataParallelKwargs = functools.partial(
    accelerate.utils.DistributedDataParallelKwargs, gradient_as_bucket_view=True)

# transformers 5 dropped the `vision_model` level from SigLIP; pi05_base still uses it,
# so without this the whole vision encoder silently stays at random init.
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

_fix_keys = PI05Policy._fix_pytorch_state_dict_keys


def _fix_keys_tf5(self, state_dict, config):
    state_dict = _fix_keys(self, state_dict, config)
    return {k.replace("vision_tower.vision_model.", "vision_tower."): v for k, v in state_dict.items()}


PI05Policy._fix_pytorch_state_dict_keys = _fix_keys_tf5

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
