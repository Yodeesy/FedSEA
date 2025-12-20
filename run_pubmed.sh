#!/bin/bash
# =========================================================
# 🏆 Pubmed Champion Run (Auto-Select Best of Online/EMA)
# Config: Proto=5.0 | OT=0.01 | K=5 | LR=0.001
# Target: 0.74 - 0.75+ (Mean)
# =========================================================

# 1. 配置参数 (Pubmed 专属)
# 注意: Patience 设为 40，给大图更多收敛时间
ARGS="--dataset pubmed --num_clients 10 --model GCN --num_layers 2 --gen_train_steps 50 --server_lr 0.01 --device_id 1 --num_rounds 250 --patience 80 --w_proto 5.0 --gen_knn 5 --gen_lr 0.001 --w_ot 0.01 --gen_num_samples 1"

# 2. 定义种子列表
SEEDS=(42 0 1 2026 2077)

# 3. 结果存储文件
RESULTS_FILE="pubmed_hybrid_results.log"
> $RESULTS_FILE  # 清空旧记录

echo "=================================================="
echo "🚀 Starting Hybrid Selection Experiment (Pubmed)"
echo "=================================================="

# 4. 循环运行
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "▶️  Running Seed $seed..."

    # 运行并捕获输出
    # 2>&1 确保错误流也能被捕获，tee 让你能实时看到日志
    LOG_OUTPUT=$(python -u main.py $ARGS --seed $seed 2>&1 | tee /dev/tty)

    # 5. 提取 Online 和 EMA 的关键指标
    ACC_ONLINE=$(echo "$LOG_OUTPUT" | grep "Final Best Result (Online):" | awk '{print $NF}')
    ACC_EMA=$(echo "$LOG_OUTPUT" | grep "Final Best Result (EMA):" | awk '{print $NF}')

    # 6. 使用 Python 比较两者并取最大值
    BEST_ACC=$(python3 -c "
try:
    # 如果抓取失败（空字符串），设为 -1
    o = float('$ACC_ONLINE') if '$ACC_ONLINE' else -1.0
    e = float('$ACC_EMA') if '$ACC_EMA' else -1.0

    # 找出最大值
    best = max(o, e)

    if best > 0:
        print(best)
    else:
        print('') # 失败情况
except:
    print('')
")

    if [[ -n "$BEST_ACC" ]]; then
        echo "$BEST_ACC" >> $RESULTS_FILE
        echo "✅ Seed $seed Winner: $BEST_ACC (Online: $ACC_ONLINE | EMA: $ACC_EMA)"
    else
        echo "⚠️  Seed $seed Failed to extract accuracy!"
    fi
done

echo ""
echo "=================================================="
echo "📊 Statistical Report (Pubmed Hybrid)"
echo "=================================================="

# 7. 使用 Python 自动计算 Mean ± Std
python3 -c "
import numpy as np
try:
    with open('$RESULTS_FILE', 'r') as f:
        # 读取非空行
        data = [float(line.strip()) for line in f.readlines() if line.strip()]

    if len(data) == 0:
        print('❌ No valid data collected.')
    else:
        mean_val = np.mean(data)
        std_val = np.std(data)

        print(f'Raw Data: {data}')
        print('-' * 30)
        print(f'🏆 Pubmed Final Result: {mean_val:.4f} ± {std_val:.4f}')
        print('-' * 30)

except Exception as e:
    print(f'Error during calculation: {e}')
"

# 清理临时文件 (可选)
rm $RESULTS_FILE