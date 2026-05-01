import os
import h5py
from filelock import FileLock
import numpy as np
import threading
import lmdb
import numpy as np
import io
import inspect

def serialize_numpy(arr: np.ndarray) -> bytes:
    """将 numpy 数组序列化为 bytes"""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def deserialize_numpy(data: bytes) -> np.ndarray:
    """将 bytes 反序列化为 numpy 数组"""
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)

class LMDBMemory:
    def __init__(self, location : str, name : str, len : int):
        self.len = len
        self.location = location
        self.name = name
        os.makedirs(location, exist_ok=True)
        # 由于多进程环境中每个进程都需要自己的环境句柄，
        # 我们不再在这里初始化环境
        # 改为在每次需要时重新打开（如果不存在）
        self.lmdb_path = os.path.join(self.location, f"{self.name}.lmdb")


    def _get_env(self):
        """获取LMDB环境，适用于多进程环境"""
        # 对于多进程场景，每个进程都应该有自己的环境句柄
        # LMDB支持多进程访问，但每个进程需要独立的Env对象
        return lmdb.open(
            self.lmdb_path,
            map_size=1024**4,
            writemap=True,
            map_async=True,
            max_readers=1024
        )

    def cache(self, key: str):
        def decorator(func):
            def wrapper(*args, **kwargs):


                # 获取函数签名
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # 提取名为index的参数
                if 'index' in bound_args.arguments:
                    index = bound_args.arguments['index']
                else:
                    raise ValueError(f"Index parameter is required for function {func.__name__}")

                buffer_key = (key + str(index)).encode('utf-8')
                # 获取LMDB环境
                env = self._get_env()
                with env.begin() as r:
                    value = r.get(buffer_key)

                if value is not None:
                    # 关闭环境并返回结果
                    env.close()
                    return deserialize_numpy(value)

                # 读取失败，需要写入
                result = func(*args, **kwargs)
                value = serialize_numpy(result)
                # 不考虑写后写，直接覆写
                with env.begin(write=True) as w:
                    w.put(buffer_key, value, overwrite=True)
                
                # 关闭环境并返回结果
                env.close()
                return result

            return wrapper
        return decorator