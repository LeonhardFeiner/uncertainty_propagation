from typing import Callable, Dict, Optional, Tuple, List, Any
import numpy as np
from torch.utils.data import Dataset, DataLoader

class EpochDataset(Dataset):
    dataset: Dataset
    data_descriptor: Dict[str, bool]
    epoch: int = 0
    transform: Callable = lambda x: x

    def __init__(
        self,
        dataset: Dataset,
        data_descriptor: Dict[str, bool],
        epoch: int = 0,
        transform: Callable = (lambda x: x),
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_descriptor = data_descriptor
        self.epoch = epoch
        self.transform = transform

    def __getitem__(self, index: int) -> Any:

        # if epoch is None:
        epoch = self.epoch

        sample = self.dataset[index]

        return self.transform(
            [
                sample[key][epoch] if is_epoch_wise else sample[key]
                for key, is_epoch_wise in self.data_descriptor
            ]
        )


class AllEpochsDataset(Dataset):
    dataset: Dataset
    data_descriptor: Dict[str, bool]
    epoch_count: int = 0
    transform: Callable = lambda x: x

    def __init__(
        self,
        dataset: Dataset,
        data_descriptor: Dict[str, bool],
        epoch_count: int = 0,
        transform: Callable = lambda x: x,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_descriptor = data_descriptor
        self.epoch_count = epoch_count
        self.transform = transform

    def select_len(self, data):
        assert len(data) >= self.epoch_count, (len(data), self.epoch_count)

        return data[: self.epoch_count]

    def repeat_len(self, data):
        return np.array([np.expand_dims(data, -1)] * self.epoch_count)
        # return np.tile(np.expand_dims(data, (0, -1)), (self.epoch_count, 1))

    def __getitem__(self, index: int) -> Any:

        sample = self.dataset[index]

        ret_val = [
            self.select_len(sample[key])
            if is_epoch_wise
            else self.repeat_len(sample[key])
            for key, is_epoch_wise in self.data_descriptor
        ]
        # print("slant", ret_val[-1])
        return self.transform(ret_val)

    def __len__(self) -> int:
        return len(self.dataset)


class RandomEpochDataset(Dataset):
    dataset: Dataset
    data_descriptor: Dict[str, bool]
    epoch_count: int = 0
    transform: Callable = lambda x: x

    def __init__(
        self,
        dataset: Dataset,
        data_descriptor: Dict[str, bool],
        epoch_count: int = 0,
        transform: Callable = (lambda x: x),
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_descriptor = data_descriptor
        self.epoch_count = epoch_count
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]

        epoch_index = np.random.randint(self.epoch_count)
        ret_val = [
            sample[key][epoch_index]
            if is_epoch_wise
            else np.expand_dims(sample[key], -1)
            for key, is_epoch_wise in self.data_descriptor
        ]
        return self.transform(ret_val)

    def __len__(self) -> int:
        return len(self.dataset)
