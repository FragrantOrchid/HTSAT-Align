import os
import numpy as np
import lmdb
import numpy as np
import io
import inspect
from pathlib import Path
from typing import List
import lmdb.aio
import asyncio

def serialize_numpy(arr: np.ndarray) -> bytes:
    """将 numpy 数组序列化为 bytes"""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def deserialize_numpy(data: bytes) -> np.ndarray:
    """将 bytes 反序列化为 numpy 数组"""
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)

def get_env(name: str, location : str = "$HOME/.cache/LMDBMemory"):
    location = os.path.expanduser(os.path.expandvars(location))
    # LMDB支持多进程访问，但每个进程需要独立的Env对象
    path = Path(location) / f"{name}.lmdb"
    return lmdb.open(
        str(path),
        map_size=1024**4,
        max_readers=1024,  # 增加最大读者数以支持更多并行进程
        writemap=True    # 使用内存映射写入，提高性能
    )

def cache(env: lmdb.Environment, unique_keys: List[str]):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if env is None:
                return func(*args, **kwargs)
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            key = [str(func.__name__)]
            for unique_key in unique_keys:
                if unique_key in bound_args.arguments:
                    key.append(str(bound_args.arguments[unique_key]))
                else:
                    raise ValueError(f"{unique_key} parameter is required for function {func.__name__}")
            key = "@".join(key)
            buffer_key = key.encode('utf-8')
            # 快速读测试
            with env.begin() as r:
                value = r.get(buffer_key)
            if value is not None:
                return deserialize_numpy(value)
            # 读取失败，需要写入
            result = func(*args, **kwargs)
            value = serialize_numpy(result)
            with env.begin(write=True) as w:
                w.put(key=buffer_key, value=value)
            return result
        return wrapper
    return decorator