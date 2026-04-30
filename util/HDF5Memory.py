import os
import h5py
from filelock import FileLock
import numpy as np
import threading

class HDF5Memory:
    def __init__(self, location : str, name : str, len : int):
        self.len = len
        self.location = location
        self.name = name
        os.makedirs(location, exist_ok=True)
        self.filepath = os.path.join(location, f"{name}.h5")
        
        # 创建一个主文件句柄
        self.file = h5py.File(self.filepath, 'a')
        # 使用FileLock保护写操作
        self.file_lock = FileLock(self.filepath)
        # 使用线程锁保护对主文件句柄的并发访问
        self.thread_lock = threading.RLock()

    def __del__(self):
        """析构函数，确保文件被正确关闭"""
        if hasattr(self, 'file') and self.file:
            self.file.close()

    def cache(self, key: str):
        def decorator(func):
            import inspect

            def wrapper(*args, **kwargs):
                # 获取函数签名
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # 提取名为index的参数
                if 'index' in bound_args.arguments:
                    index = int(bound_args.arguments['index'])
                else:
                    raise ValueError(f"Index parameter is required for function {func.__name__}")
                
                # 检查索引是否在有效范围内
                if index < 0 or index >= self.len:
                    raise IndexError(f"Index {index} is out of bounds for cache of length {self.len}")
                
                
                
                
                
                # 首先尝试快速读取（使用线程锁）
                with self.thread_lock:
                    # 如果数据集存在且数据已缓存，则直接返回
                    if key in self.file:
                        dataset = self.file[key]
                        cached_data = dataset[index]
                        if not np.isnan(cached_data).all():
                            return cached_data
                        
                        # 数据未缓存，需要计算并写入
                        # 升级到文件锁以进行写操作
                        with self.file_lock:
                            # 再次检查，以防其他线程在此期间写入了数据
                            if key in self.file:
                                dataset = self.file[key]
                                cached_data = dataset[index]
                                if not np.isnan(cached_data).all():
                                    # 其他线程已经写入了数据，直接返回
                                    return cached_data
                                else:
                                    # 数据仍未缓存，需要计算并写入
                                    true_data = func(*args, **kwargs)
                                    dataset[index] = np.asarray(true_data, dtype=np.float32)
                                    return true_data
                            else:
                                # 数据集不存在，需要创建
                                true_data = func(*args, **kwargs)  # 计算初始值以确定形状
                                dataset = self.file.create_dataset(
                                    name=key,
                                    shape=(self.len, *true_data.shape),    # 初始第一维为总长度，其余维度根据首次调用确定
                                    dtype=np.float32,
                                    chunks=(min(10, self.len), *true_data.shape),
                                    fillvalue=np.float32(np.nan)
                                )
                                dataset[index] = np.asarray(true_data, dtype=np.float32)
                                return true_data
                    else:
                        # 数据集不存在，需要在文件锁下创建
                        with self.file_lock:
                            # 再次检查数据集是否存在（双重检查）
                            if key in self.file:
                                # 另一个线程创建了数据集，继续使用线程锁逻辑
                                dataset = self.file[key]
                                cached_data = dataset[index]
                                if not np.isnan(cached_data).all():
                                    return cached_data
                                else:
                                    # 数据仍未缓存，需要计算并写入
                                    true_data = func(*args, **kwargs)
                                    dataset[index] = np.asarray(true_data, dtype=np.float32)
                                    return true_data
                            else:
                                # 确认数据集不存在，创建它
                                true_data = func(*args, **kwargs)  # 计算初始值以确定形状
                                dataset = self.file.create_dataset(
                                    name=key,
                                    shape=(self.len, *true_data.shape),    # 初始第一维为总长度，其余维度根据首次调用确定
                                    dtype=np.float32,
                                    chunks=(min(10, self.len), *true_data.shape),
                                    fillvalue=np.float32(np.nan)
                                )
                                dataset[index] = np.asarray(true_data, dtype=np.float32)
                                return true_data
            return wrapper
        return decorator