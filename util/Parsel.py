import parselmouth
import numpy as np
import torch
import parselmouth
import numpy as np
import torch.nn.functional as F
np.set_printoptions(
    threshold=np.inf,
    precision=4,
    linewidth=np.inf
)
vowel_ref = {
    'i' :   (240,2400),
    'y' :   (235,2100),
    'e' :   (390,2300),
    'ø' :   (370,1900),
    'ɛ' :   (610,1900),
    'œ':   (585,1710),
    'a' :   (850,1610),
    'æ':    (820,1530),
    'ɑ' :   (750,940),
    'ɒ' :   (700,760),
    'ʌ' :   (600,1170),
    'ɔ' :   (500,700),
    'ɤ' :   (460,1310),
    'o' :   (360,640),
    'ɯ' :   (300,1390),
    'u' :   (250,595)
}

def softmax(x, temperature=1.0):
    x = np.array(x)
    x = x / temperature  # 温度参数
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_vowel_prob(f1, f2, gender='male', age_adult=True):
    """
    输入: 
        f1, f2 - 共振峰频率 (Hz)
        gender - 'male', 'female', 'child'
        age_adult - 如果为False且gender为'male'，则视为男孩
    返回: 
        字典，键为元音，值为概率
    """
    adjusted_ref = {}
    for vowel, (ref_f1, ref_f2) in vowel_ref.items():
        adj_f1, adj_f2 = ref_f1, ref_f2
        adjusted_ref[vowel] = (adj_f1, adj_f2)
    
    # 计算观测值到各参考点的欧氏距离
    distances = {}
    for vowel, (ref_f1, ref_f2) in adjusted_ref.items():
        dist = np.sqrt((f1 - ref_f1)**2 + (f2 - ref_f2)**2)
        distances[vowel] = dist
    
    # 将距离转换为概率（距离越小概率越高）
    dist_array = np.array(list(distances.values()))
    # 使用负距离，因为距离越小越可能
    probs = softmax(-dist_array,temperature=100)
    
    return {vowel: prob for vowel, prob in zip(distances.keys(), probs)}
def get_matrix(filename):
    # 读取音频文件
    sound = parselmouth.Sound(filename)

    # 提取共振峰（前3个）
    formants = sound.to_formant_burg(max_number_of_formants=3)

    # 获取元音列表
    vowels = list(vowel_ref.keys())
    num_vowels = len(vowels)

    # 创建时间点数组
    time_points = np.arange(1.0/320, 1.0, 1.0/160)
    num_frames = len(time_points)
    # print(f"num_frames {num_frames}")

    # 创建结果矩阵 (timeframes × numofvowel)
    prob_matrix = np.zeros((num_frames, num_vowels))

    for i, time_point in enumerate(time_points):
        f1 = formants.get_value_at_time(1, time_point)
        f2 = formants.get_value_at_time(2, time_point)

        # 获取该时间点的元音概率
        probs = get_vowel_prob(f1, f2)

        # 将概率按元音顺序存储到矩阵中
        for j, vowel in enumerate(vowels):
            prob_matrix[i, j] = probs[vowel]

    # 输出结果矩阵
    return np.nan_to_num(prob_matrix,0.0)




def get_vowel_prob_matrix(filename: str, sigma: float = 100.0, device: str = 'cpu') -> torch.Tensor:
    """
    从音频文件生成元音概率矩阵
    
    参数:
        filename: 音频文件路径
        sigma: 相似度衰减参数
        device: 计算设备 ('cpu' 或 'cuda')
        
    返回:
        PyTorch 张量，形状为 [num_frames, num_vowels]，表示每个时间点的元音概率
    """
    # 读取音频文件
    sound = parselmouth.Sound(filename)
    
    # 提取共振峰（前3个）
    formants = sound.to_formant_burg(max_number_of_formants=3)
    
    # 获取时间点数组
    duration = sound.get_total_duration()
    time_points = np.arange(0.5 / 160, duration, 1.0 / 160)
    num_frames = len(time_points)
    
    # 准备参考点数据
    vowels = list(vowel_ref.keys())
    num_vowels = len(vowels)
    ref_points = torch.tensor([vowel_ref[v] for v in vowels], 
                             dtype=torch.float32, 
                             device=device)
    
    # 提取所有时间点的 F1 和 F2
    f1_values = np.array([formants.get_value_at_time(1, t) for t in time_points])
    f2_values = np.array([formants.get_value_at_time(2, t) for t in time_points])
    
    # 处理 NaN 值（设置为参考点范围外的值）
    f1_mean = np.nanmean(f1_values)
    f2_mean = np.nanmean(f2_values)
    f1_values = np.nan_to_num(f1_values, nan=f1_mean)
    f2_values = np.nan_to_num(f2_values, nan=f2_mean)
    
    # 创建查询点张量
    query_points = torch.tensor(np.column_stack((f1_values, f2_values)),
                              dtype=torch.float32,
                              device=device)
    
    # 计算所有点对之间的距离
    # query_points: [num_frames, 2], ref_points: [num_vowels, 2]
    query_exp = query_points.unsqueeze(1)  # [num_frames, 1, 2]
    ref_exp = ref_points.unsqueeze(0)      # [1, num_vowels, 2]
    
    distances = torch.norm(query_exp - ref_exp, dim=2)  # [num_frames, num_vowels]
    
    # 将距离转换为相似度（使用高斯核）
    similarities = torch.exp(-distances**2 / (2 * sigma**2))  # [num_frames, num_vowels]
    
    # 归一化得到概率分布
    probabilities = similarities / similarities.sum(dim=1, keepdim=True)
    
    # 处理可能出现的NaN（当所有相似度为0时）
    nan_mask = torch.isnan(probabilities).any(dim=1)
    if nan_mask.any():
        # 对于无效点，设置为均匀分布
        uniform_probs = torch.ones(num_vowels, device=device) / num_vowels
        probabilities[nan_mask] = uniform_probs
    
    return probabilities
    