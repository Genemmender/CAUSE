import base64
import hashlib
import json
import os
import random
import string
import sys
from typing import Optional

import numpy as np
import torch


def combine_config(config: dict, **kwargs):
    """
    将额外传入的 key-value 更新进 config 字典中，前提是 config 中不存在该 key。
    用于参数补全或默认值设置。
    """
    for k, v in kwargs.items():
        if k not in config:
            config[k] = v
    return config


def seeding(seed=2023):
    """
    设置所有相关库的随机种子，确保实验的可复现性。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # 如果使用 TensorFlow，可启用以下行
    # tensorflow.random.set_seed(seed)


def argparse(arguments=None):
    """
    一个简易命令行参数解析器，将形如 --key value 的参数转换成字典。
    支持自动类型转换：int、float、bool、None。
    """
    arguments = arguments or sys.argv[1:]
    kwargs = {}
    key: Optional[str] = None

    for arg in arguments:
        if key is not None:
            kwargs[key] = arg  # 当前是值，保存到对应 key
            key = None
        else:
            assert arg.startswith('--')  # key 必须以 -- 开头
            key = arg[2:]

    # 类型转换
    for key, value in kwargs.items():  # type: str, str
        if value == 'null':
            kwargs[key] = None
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            kwargs[key] = int(value)
        elif value.lower() == 'true':
            kwargs[key] = True
        elif value.lower() == 'false':
            kwargs[key] = False
        else:
            try:
                kwargs[key] = float(value)
            except ValueError:
                pass  # 保持为字符串
    return kwargs


def get_signature(data: dict):
    """
    生成数据的短签名：对 JSON 格式的字典进行 MD5 -> base64 编码后截取前8位。
    可用于生成唯一但可读性强的标识符（如实验名）。
    """
    canonical_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    md5_digest = hashlib.md5(canonical_str.encode('utf-8')).digest()
    b64_str = base64.urlsafe_b64encode(md5_digest).decode('utf-8').rstrip('=')
    return b64_str[:8]  # 截取前8位作为签名


def get_exp_signature(data, model, exp):
    """
    基于 data, model, exp 三个字段生成实验签名，用于唯一标识一次实验配置。
    """
    configuration = {
        'data': data,
        'model': model,
        'exp': exp,
    }
    return get_signature(configuration)


def get_random_string(length=6):
    """
    生成指定长度的随机字符串，包含大小写字母和数字。
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def package_require(name, version=None):
    """
    检查指定 Python 包是否安装，并可选要求最低版本。
    如果未安装或版本不满足要求，抛出友好错误提示。
    """
    from importlib.metadata import version as get_version, PackageNotFoundError

    try:
        installed_version = get_version(name)
    except PackageNotFoundError:
        raise ImportError(f"Package {name} not found. Please use 'pip install {name}' to install.")

    if version is not None:
        if installed_version < version:
            raise ImportError(
                f"Package {name} version is {installed_version}, "
                f"but require version {version}. "
                f"Please use 'pip install {name} -U' to upgrade."
            )
