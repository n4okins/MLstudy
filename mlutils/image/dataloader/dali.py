import ctypes
import torch
from nvidia.dali.backend import TensorCPU, TensorGPU
from nvidia.dali.types import DALIDataType

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