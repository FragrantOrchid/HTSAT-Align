import parselmouth
import numpy as np
np.set_printoptions(
    threshold=np.inf,
    precision=4,
    linewidth=np.inf
)
# 元音参考表（成年男性基准）
vowel_ref = {
    'i': (240, 2400),   # beet
    'ɪ': (360, 2100),   # bit
    'e': (530, 1850),   # bet
    'æ': (660, 1720),   # bat
    'ɑ': (730, 1090),   # father
    'ɔ': (570, 840),    # bought
    'ʊ': (440, 1020),   # book
    'u': (300, 870),    # boot
    'ʌ': (640, 1190),   # but
    'ɝ': (490, 1350),   # bird
    'ə': (500, 1450)    # about
}

def adjust_for_speaker(f1, f2, gender='male', age_adult=True):
    """根据说话人类型调整参考值"""
    if gender == 'female':
        f1 = f1 * 1.2
        f2 = f2 * 1.1
    elif gender == 'child':
        f1 = f1 * 1.3
        f2 = f2 * 1.3
    elif gender == 'male' and not age_adult:
        f1 = f1 * 1.2
        f2 = f2 * 1.2
    return f1, f2

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
        adj_f1, adj_f2 = adjust_for_speaker(ref_f1, ref_f2, gender, age_adult)
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
    print(f"num_frames {num_frames}")

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
    return np.nan_to_zero(prob_matrix)