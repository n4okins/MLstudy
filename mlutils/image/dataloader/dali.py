import ctypes
from functools import partial
from logging import Logger, getLogger
from typing import Any, Callable

import torch
from torchvision.datasets import ImageFolder
from nvidia.dali import fn as dali_fn
from nvidia.dali.backend import TensorCPU, TensorGPU
from nvidia.dali.pipeline import Pipeline as DALIPipeline
from nvidia.dali.types import DALIDataType, DALIImageType

DALIDataType2TorchDataType = {
    DALIDataType.FLOAT: torch.float32,
    DALIDataType.FLOAT64: torch.float64,
    DALIDataType.FLOAT16: torch.float16,
    DALIDataType.UINT8: torch.uint8,
    DALIDataType.INT8: torch.int8,
    DALIDataType.BOOL: torch.bool,
    DALIDataType.INT16: torch.int16,
    DALIDataType.INT32: torch.int32,
    DALIDataType.INT64: torch.int64,
}


def dali_to_torch_tensor(
    dali_tensor: TensorCPU | TensorGPU, device_id=None
) -> torch.Tensor:
    stream = None
    device = torch.device("cpu")
    if isinstance(dali_tensor, TensorGPU):
        device = torch.device("cuda", index=device_id or 0)
        stream = torch.cuda.current_stream(device=device)
    torch_tensor = torch.empty(
        dali_tensor.shape(),
        dtype=DALIDataType2TorchDataType[dali_tensor.dtype],
        device=device,
    )
    pointer = ctypes.c_void_p(torch_tensor.data_ptr())
    if device.type == "cuda":
        stream = stream if stream is None else ctypes.c_void_p(stream.cuda_stream)
        dali_tensor.copy_to_external(pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(pointer)

    return torch_tensor


class DALIFromImageFolder(DALIPipeline):
    def default_collate_fn(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels

    def __init__(
        self,
        dataset: ImageFolder,
        transform: Callable = lambda x: x,
        target_transform: Callable = lambda x: x,
        batch_size: int = 1,
        num_workers: int = 12,
        shuffle: bool = True,
        device_id: int = 0,
        seed: int | None = None,
        logger: Logger = getLogger(__name__),
        collate_fn: Callable[[Any], Any] | None = None,
        *args,
        **kwargs,
    ):
        super(DALIFromImageFolder, self).__init__(batch_size, num_workers, device_id)
        self.dataset = dataset

        self.transform = transform
        self.target_transform = target_transform

        self.file_pathes, self.labels = zip(*self.dataset.samples)
        self.collate_fn = collate_fn or self.default_collate_fn

        self.input = dali_fn.readers.file(
            files=self.file_pathes,
            labels=self.labels,
            name="Reader",
            random_shuffle=shuffle,
            seed=seed,
        )
        self.decode = partial(
            dali_fn.decoders.image, device="cpu", output_type=DALIImageType.RGB
        )
        self.build()

        self._index = 0

    def define_graph(self):
        images, labels = self.input
        images = self.decode(images)
        return (images, labels)

    @property
    def index(self):
        return self._index

    def __len__(self):
        return len(self.dataset) // self.max_batch_size + 1

    def __iter__(self):
        for i in range(len(self)):
            images, labels = self.run()
            images = list(map(dali_to_torch_tensor, images))
            labels = list(map(dali_to_torch_tensor, labels))
            if self.transform:
                images = list(map(self.transform, images))
            if self.target_transform:
                labels = list(map(self.target_transform, labels))
            images, labels = self.collate_fn(list(zip(images, labels)))
            yield images, labels
