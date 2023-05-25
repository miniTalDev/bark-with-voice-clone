# From https://github.com/huggingface/transformers/blob/e45e756d22206ca8fa9fb057c8c3d8fa79bf81c6/src/transformers/utils/bitsandbytes.py

import warnings
import sys
import importlib.util
from copy import deepcopy
import copy
import json
import os
from dataclasses import dataclass

from typing import Any, Tuple, Union, Dict

from packaging import version

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib_metadata.version(pkg_name)
            package_exists = True
        except importlib_metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

_accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)
_bitsandbytes_available = _is_package_available("bitsandbytes")
_torch_available, _torch_version = _is_package_available("torch", return_version=True)

def is_accelerate_available(check_partial_state=False):
    if check_partial_state:
        return _accelerate_available and version.parse(_accelerate_version) >= version.parse("0.19.0")
    return _accelerate_available

def is_bitsandbytes_available():
    return _bitsandbytes_available

def is_torch_available():
    return _torch_available

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import find_tied_parameters


def set_module_quantized_tensor_to_device(module, tensor_name, device, value=None, fp16_statistics=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function). The
    function is adapted from `set_module_tensor_to_device` function from accelerate that is adapted to support the
    class `Int8Params` from `bitsandbytes`.

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        fp16_statistics (`torch.HalfTensor`, *optional*):
            The list of fp16 statistics to set on the module, used for serialization.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    is_4bit = False
    is_8bit = False
    if is_buffer or not is_bitsandbytes_available():
        is_8bit = False
        is_4bit = False
    else:
        is_4bit = hasattr(bnb.nn, "Params4bit") and isinstance(module._parameters[tensor_name], bnb.nn.Params4bit)
        is_8bit = isinstance(module._parameters[tensor_name], bnb.nn.Int8Params)

    if is_8bit or is_4bit:
        param = module._parameters[tensor_name]
        if param.device.type != "cuda":
            if value is None:
                new_value = old_value.to(device)
            elif isinstance(value, torch.Tensor):
                new_value = value.to("cpu")
                if value.dtype == torch.int8:
                    is_8bit_serializable = version.parse(importlib_metadata.version("bitsandbytes")) > version.parse(
                        "0.37.2"
                    )
                    if not is_8bit_serializable:
                        raise ValueError(
                            "Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. "
                            "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                        )
            else:
                new_value = torch.tensor(value, device="cpu")

            kwargs = old_value.__dict__
            if is_8bit:
                new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(device)
            elif is_4bit:
                new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(device)

            module._parameters[tensor_name] = new_value
            if fp16_statistics is not None:
                setattr(module.weight, "SCB", fp16_statistics.to(device))

    else:
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        else:
            new_value = nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value


def replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `LLM.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes`

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Int8 mixed-precision matrix decomposition works by separating a
    matrix multiplication into two streams: (1) and systematic feature outlier stream matrix multiplied in fp16
    (0.01%), (2) a regular stream of int8 matrix multiplication (99.9%). With this method, int8 inference with no
    predictive degradation is possible for very large models (>=176B parameters).

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `Linear8bitLt`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    if quantization_config.quantization_method() == "llm_int8":
                        model._modules[name] = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            module.bias is not None,
                            has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                            threshold=quantization_config.llm_int8_threshold,
                        )
                    else:
                        if (
                            quantization_config.llm_int8_skip_modules is not None
                            and name in quantization_config.llm_int8_skip_modules
                        ):
                            pass
                        else:
                            model._modules[name] = bnb.nn.Linear4bit(
                                module.in_features,
                                module.out_features,
                                module.bias is not None,
                                quantization_config.bnb_4bit_compute_dtype,
                                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                                quant_type=quantization_config.bnb_4bit_quant_type,
                            )
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
            )
    return model


# For backward compatibility
def replace_8bit_linear(*args, **kwargs):
    warnings.warn(
        "`replace_8bit_linear` will be deprecated in a future version, please use `replace_with_bnb_linear` instead",
        FutureWarning,
    )
    return replace_with_bnb_linear(*args, **kwargs)


# For backward compatiblity
def set_module_8bit_tensor_to_device(*args, **kwargs):
    warnings.warn(
        "`set_module_8bit_tensor_to_device` will be deprecated in a future version, please use `set_module_quantized_tensor_to_device` instead",
        FutureWarning,
    )
    return set_module_quantized_tensor_to_device(*args, **kwargs)


def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = list(tied_params.values())
    else:
        tied_keys = sum([x[1:] for x in tied_params], [])
    has_tied_params = len(tied_keys) > 0

    # Check if it is a base model
    is_base_model = not hasattr(model, model.base_model_prefix)

    # Ignore this for base models (BertModel, GPT2Model, etc.)
    if (not has_tied_params) and is_base_model:
        return []

    # otherwise they have an attached head
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]

    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = tied_keys + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names

#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



if is_torch_available():
    import torch


@dataclass
class BitsAndBytesConfig:
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`List[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input time. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`, {fp4, fn4}, defaults to `fp4`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `fn4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        **kwargs,
    ):
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.llm_int8_threshold, float):
            raise ValueError("llm_int8_threshold must be a float")

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise ValueError("llm_int8_skip_modules must be a list of strings")
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise ValueError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise ValueError("llm_int8_has_fp16_weight must be a boolean")

        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise ValueError("bnb_4bit_compute_dtype must be torch.dtype")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise ValueError("bnb_4bit_quant_type must be a string")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise ValueError("bnb_4bit_use_double_quant must be a boolean")

        if self.load_in_4bit and not version.parse(importlib_metadata.version("bitsandbytes")) >= version.parse(
            "0.39.0"
        ):
            raise ValueError(
                "4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version"
            )

    def is_quantizable(self):
        r"""
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs, **kwargs):
        """
        Instantiates a [`BitsAndBytesConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`BitsAndBytesConfig`]: The configuration object instantiated from those parameters.
        """

        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `BitsAndBytesConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """

        output = copy.deepcopy(self.__dict__)
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]

        return output