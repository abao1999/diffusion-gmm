from typing import Union

import torch
from torch.utils.data import Dataset


class MultiClassSubset(Dataset):
    """
    Wrap a subset of a dataset to apply a mapping of targets (class labels) to
    user-specified multi-class classification labels
    """

    def __init__(
        self,
        subset,
        class_to_index,
        device: Union[torch.device, str] = "cpu",
    ):
        self.subset = subset
        self.class_to_index = class_to_index
        self.num_classes = len(class_to_index)
        self.device = device

    def __getitem__(self, index):
        data, target = self.subset[index]
        label = self.class_to_index[target]

        return data.to(self.device), torch.tensor(label, dtype=torch.float).to(
            self.device
        )

    def __len__(self):
        return len(self.subset)


class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if "cuda" in str(self.device)
            else None
        )
        self.dataset = self.loader.dataset
        self.reset()  # Initialize the iterator and prefetch the first batch

    def reset(self):
        # Reset the iterator for the DataLoader, which will have reshuffled data if `shuffle=True`
        self.loader_iter = iter(self.loader)
        self.prefetch()

    def prefetch(self):
        try:
            self.next_input, self.next_target = next(self.loader_iter)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        if self.stream:
            with torch.cuda.stream(self.stream):  # type: ignore
                self.next_input = self.next_input.to(self.device, non_blocking=True)
                self.next_target = self.next_target.to(self.device, non_blocking=True)
        else:
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)

    def next(self):
        if self.stream:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.prefetch()
        return input, target

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.next_input is None:
            raise StopIteration
        return self.next()
