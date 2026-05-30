#!/bin/bash
#SBATCH --job-name="HTAST-Align"
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --nodelist=gpu2

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

#SBATCH --output=/home/u220110626/.slurm/slurm-%j.out
#SBATCH --error=/home/u220110626/.slurm/slurm-%j.err


nvidia-smi
source ~/.bashrc
conda activate HLHTSAT-v4
export TMPDIR=~/.tmp
export PYTHONDONTWRITEBYTECODE=1
dataset=speechcommand-v2
report_name="speechcommand-v2-部分冻结"
# 数据集配置文件
config_file="dataset.conf"
if [ ! -f "$config_file" ]; then
    echo "配置文件 $config_file 不存在"
    exit 1
fi
found=false
while IFS= read -r line; do
    if [[ $line == \[$dataset\] ]]; then
        found=true
        continue
    fi
    if [[ $line == \[*\] ]]; then
        found=false
        continue
    fi
    if $found; then
        line=$(echo "$line" | xargs)
        eval "$line"
    fi
done < "$config_file"


exp_name="[$(date +"%F-%T")]"
exp_dir=./export/${exp_name}
mkdir -p ~/data/Export/${exp_name}
ln -sf ~/data/Export/${exp_name}  $exp_dir


# 从部分冻结的模型开始训练
mode="train"
ckpt_path="./ckpt/htast-align-after-pretraining.ckpt" 
# 测试整体微调后的模型
# mode="test"
# ckpt_path="./ckpt/htast-align-after-partial-freezing.ckpt"

command="python -u train.py \
-train_file ${train_file} \
-val_file ${val_file} \
-test_file ${test_file} \
-label_csv ${label_csv} \
-class_num ${class_num} \
-sound_length ${sound_length} \
-export_path ${exp_dir} \
-report_name ${report_name} \
-mode ${mode} \
-ckpt_path ${ckpt_path}"


export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo $command
# 执行命令并记录日志
N_PROC=$(nproc) srun -u $command 2>&1 | tee "$exp_dir/run.log"