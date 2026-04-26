import torch
import torch.nn as nn
import torch.nn.functional as F
class SGMWithGEModule(nn.Module):
    """
    模仿原始SGM模型结构，特别是加入GE（Gradient Estimation）模块的特性
    将(Batch, num_class, hidden_channel)形状的张量转换为(batch, num_class)的模块，
    同时允许前一个标签对后面的标签产生影响。
    """
    
    def __init__(self, hidden_channel, num_class, tau=1.0, dropout=0.1):
        super(SGMWithGEModule, self).__init__()
        
        self.num_class = num_class
        self.hidden_channel = hidden_channel
        self.tau = tau  # 温度参数，用于softmax
        
        # 用于模拟标签嵌入的参数矩阵
        self.label_embeddings = nn.Parameter(torch.randn(num_class, hidden_channel))
        
        # GE模块相关参数
        self.ge_proj1 = nn.Linear(hidden_channel, hidden_channel)
        self.ge_proj2 = nn.Linear(hidden_channel, hidden_channel)
        
        # RNN层，用于处理标签间的序列依赖关系
        self.rnn = nn.GRU(
            input_size=hidden_channel,  # 输入维度
            hidden_size=hidden_channel,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )
        
        # 线性层，将隐藏状态转换为标签预测分数
        self.output_layer = nn.Linear(hidden_channel, 1)
        
        # Softmax层，用于计算概率分布
        self.softmax = nn.Softmax(dim=1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_class, hidden_channel] 输入张量
        
        Returns:
            output: [batch_size, num_class] 标签预测分数
        """
        batch_size, num_class, hidden_channel = x.shape
        
        # 确保输入形状正确
        assert num_class == self.num_class and hidden_channel == self.hidden_channel
        
        # 初始化输出为零
        prev_output = None
        
        outputs = []
        
        # 逐个处理每个标签，模仿原始SGM的序列生成过程
        for i in range(num_class):
            # 当前标签的特征
            current_features = x[:, i, :]  # [batch_size, hidden_channel]
            
            # 如果是第一个标签或没有前序输出，则初始化
            if prev_output is None:
                prev_output = current_features.new_zeros(batch_size, self.num_class)
            
            # 使用前序输出计算全局嵌入（GE模块的核心思想）
            probs = self.softmax(prev_output / self.tau)  # [batch_size, num_class]
            # 使用当前标签的嵌入来计算平均嵌入
            emb_avg = torch.matmul(probs, self.label_embeddings)  # [batch_size, hidden_channel]
            
            # GE模块的门控机制
            H = torch.sigmoid(self.ge_proj1(current_features) + self.ge_proj2(emb_avg))
            emb_glb = H * current_features + (1 - H) * emb_avg
            
            # 使用处理后的特征作为RNN的输入
            rnn_input = emb_glb  # [batch_size, hidden_channel]
            
            # 如果是第一次，需要初始化隐藏状态
            if i == 0:
                # 使用第一个标签的特征初始化隐藏状态
                h = torch.tanh(rnn_input).unsqueeze(0)  # [1, batch_size, hidden_channel]
                rnn_output, h = self.rnn(rnn_input.unsqueeze(1), h)
            else:
                rnn_output, h = self.rnn(rnn_input.unsqueeze(1), h)
            
            # 将RNN输出转换为当前标签的预测分数
            label_score = self.output_layer(self.dropout(rnn_output.squeeze(1)))  # [batch_size, 1]
            
            # 更新prev_output，只更新当前标签位置的值
            prev_output = torch.cat([
                prev_output[:, :i], 
                label_score.squeeze(-1).unsqueeze(1), 
                prev_output[:, i+1:]
            ], dim=1)
            
            outputs.append(label_score)
        
        # 拼接所有标签的预测结果
        output = torch.cat(outputs, dim=1)  # [batch_size, num_class]

        return output