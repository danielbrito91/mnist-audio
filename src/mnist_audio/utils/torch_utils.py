from typing import Union

import torch


def get_device(device: Union[torch.device, str, None] = None) -> torch.device:  # type: ignore[arg-type]
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
