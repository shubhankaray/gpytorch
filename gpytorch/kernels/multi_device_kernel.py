#!/usr/bin/env python3

import concurrent.futures
import torch
from torch.cuda._utils import _get_device_index
from torch.nn.parallel import DataParallel
from .kernel import Kernel
from ..lazy import CatLazyTensor, lazify
from .. import settings


class MultiDeviceKernel(DataParallel, Kernel):
    r"""
    Allocates the covariance matrix on distributed devices, e.g. multiple GPUs.

    Args:
        - :attr:`base_kernel`: Base kernel to distribute
        - :attr:`device_ids`: list of `torch.device` objects to place kernel chunks on
        - :attr:`output_device`: Device where outputs will be placed
    """

    def __init__(self, base_kernel, device_ids, output_device=None,
                 create_cuda_context=True, **kwargs):
        DataParallel.__init__(self,
                              module=base_kernel,
                              device_ids=device_ids,
                              output_device=output_device,
                              dim=-2)
        self.output_device = output_device if output_device else device_ids[0]

        # Warm up the GPUs to reduce replicate and scatter time
        self.module = self.module.to(self.output_device)
        self.replicas = self.replicate(self.module, self.device_ids)

        self.__cached_x1 = torch.empty(1)
        self.__cached_x2 = torch.empty(1)

        num_devices = len(self.device_ids)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_devices)
        # Start the threads now instead of later when needed
        for _ in range(num_devices):
            self.executor._adjust_thread_count()

    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            return self.module.forward(x1, x2, diag=True, **kwargs).to(self.output_device)

        if not x1.device == self.__cached_x1.device or not torch.equal(x1, self.__cached_x1):
            self._x1_scattered, self._kwargs = self.scatter((x1,), kwargs, self.device_ids)
            self.__cached_x1 = x1

        if not x2.device == self.__cached_x2.device or not torch.equal(x2, self.__cached_x2):
            self._x2_subs = [x2.to(x1_[0].device, non_blocking=True) for x1_ in self._x1_scattered]
            self.__cached_x2 = x2

        inputs = tuple((x1_[0], x2_) for x1_, x2_ in zip(self._x1_scattered, self._x2_subs))

        if not self.device_ids:
            return self.module.forward(*inputs, **self._kwargs)

        if len(self.device_ids) == 1:
            return self.module.forward(*inputs[0], **self._kwargs[0])

        # Can't cache the replication because the base kernel module can change every time (e.g. param updates)
        replicate_start = time.time()
        self.replicas = self.replicate(self.module, self.device_ids)

        with settings.lazily_evaluate_kernels(False):
            outputs = self.parallel_apply(self.replicas, inputs, self._kwargs, self.device_ids[:len(self.replicas)])

        return self.gather(outputs, self.output_device)

    def gather(self, outputs, output_device):
        res = CatLazyTensor(*[lazify(o) for o in outputs], dim=self.dim, output_device=self.output_device)
        return res

    def size(self, x1, x2):
        return self.module.size(x1, x2)

    def parallel_apply(self, modules, inputs, kwargs_tup=None, devices=None):
        """
        Adapts torch.nn.paralle.parallel_apply to use concurrent.futures
        """
        assert len(modules) == len(inputs)
        if kwargs_tup is not None:
            assert len(modules) == len(kwargs_tup)
        else:
            kwargs_tup = ({},) * len(modules)
        if devices is not None:
            assert len(modules) == len(devices)
        else:
            devices = [None] * len(modules)
        devices = list(map(lambda x: _get_device_index(x, True), devices))
        grad_enabled = torch.is_grad_enabled()

        def _worker(i, module, input, kwargs, device=None):
            torch.set_grad_enabled(grad_enabled)
            if device is None:
                device = get_a_var(input).get_device()
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                return module(*input, **kwargs)

        if len(modules) > 1:
            results = self.executor.map(_worker, range(len(devices)), modules,inputs, kwargs_tup)
        else:
            results = [_worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])]
        return results


def get_a_var(obj):
   if isinstance(obj, torch.Tensor):
       return obj

   if isinstance(obj, list) or isinstance(obj, tuple):
       for result in map(get_a_var, obj):
           if isinstance(result, torch.Tensor):
               return result
   if isinstance(obj, dict):
       for result in map(get_a_var, obj.items()):
           if isinstance(result, torch.Tensor):
               return result
   return None
