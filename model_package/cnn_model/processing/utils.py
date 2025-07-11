import os
import random
import re
from typing import List, Tuple

import numpy as np
import torch

from cnn_model.config.core import config


def set_seed(seed: int = config.model_conf.random_state):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fetch_epoch_id(model_fnames: List[str]) -> Tuple[int, List[int]]:

    all_epochs = []
    highest_epoch = 0
    for fname in model_fnames:
        match = re.match(
            r"^cnn_model_output_epoch_(\d+)_v[\d\.]+\.pth$", os.path.basename(fname)
        )
        if match:
            epoch_id = int(match.group(1))
            all_epochs.append(epoch_id)
            if highest_epoch < epoch_id:
                highest_epoch = epoch_id

    return highest_epoch, all_epochs
