def load_part_of_state_dict(model, state_dict, strict=False):
    """
    加载部分模型状态字典，对于不匹配的参数保持随机初始化
    """
    model_state_dict = model.state_dict()
    
    # 创建一个新的状态字典，只包含匹配的参数
    filtered_state_dict = {}
    unmatched_keys = []
    
    for key, param in state_dict.items():
        if key in model_state_dict:
            # 检查参数形状是否匹配
            if param.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = param
                print(f"加载匹配的参数: {key}, 形状：{param.shape}")
            else:
                print(f"跳过不匹配的参数: {key}, "
                      f"检查点形状: {param.shape}, 当前模型形状: {model_state_dict[key].shape}")
                unmatched_keys.append(key)
        elif not strict:
            # 如果不是严格模式，记录未找到的键
            unmatched_keys.append(key)
        else:
            raise KeyError(f"Unexpected key(s) in state_dict: {key}")
    
    # 将过滤后的状态字典加载到模型中
    model.load_state_dict(filtered_state_dict, strict=strict)
    
    # 打印加载结果
    matched_keys = list(filtered_state_dict.keys())
    print(f"成功加载 {len(matched_keys)} 个参数")
    if unmatched_keys:
        print(f"跳过 {len(unmatched_keys)} 个参数: {unmatched_keys}")
    
    return model