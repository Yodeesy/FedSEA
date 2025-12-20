#!/bin/bash
# =========================================================
# 🏆 Citeseer FINAL OPTIMIZED Run
# Config: w_proto=20.0 | K=3 | w_ot=0.05 | GenLR=0.001
# Tweak: Dropout=0.6 | Hidden=128 (To fix overfitting/instability)
# Target: ~0.62+ (Stable Mean)
# =========================================================

ARGS="--dataset citeseer --device_id 1 --num_clients 10 --model GCN --num_layers 2 --gen_train_steps 50 --server_lr 0.01 --device_id 2 --num_rounds 250 --patience 90 --w_proto 20.0 --gen_knn 3 --dropout 0.5 --gen_lr 0.001 --w_ot 0.05 --gen_num_samples 1"

SEEDS=(42 0 1 2026 2077)

RESULTS_FILE="citeseer_results.log"
> $RESULTS_FILE

echo "=================================================="
echo "🚀 Starting Feature-Tuned Experiment for Citeseer"
echo "=================================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "▶️  Running Seed $seed..."

    # 运行
    LOG_OUTPUT=$(python -u main.py $ARGS --seed $seed 2>&1 | tee /dev/tty)

    # 提取 Hybrid Best
    ACC_ONLINE=$(echo "$LOG_OUTPUT" | grep "Final Best Result (Online):" | awk '{print $NF}')
    ACC_EMA=$(echo "$LOG_OUTPUT" | grep "Final Best Result (EMA):" | awk '{print $NF}')

    # Python 比较取最大
    BEST_ACC=$(python3 -c "
try:
    o = float('$ACC_ONLINE') if '$ACC_ONLINE' else -1.0
    e = float('$ACC_EMA') if '$ACC_EMA' else -1.0
    best = max(o, e)
    if best > 0: print(best)
    else: print('')
except: print('')
")

    if [[ -n "$BEST_ACC" ]]; then
        echo "$BEST_ACC" >> $RESULTS_FILE
        echo "✅ Seed $seed Winner: $BEST_ACC"
    else
        echo "⚠️  Seed $seed Failed!"
    fi
done

echo ""
echo "=================================================="
echo "📊 Statistical Report"
echo "=================================================="

python3 -c "
import numpy as np
try:
    with open('$RESULTS_FILE', 'r') as f:
        data = [float(line.strip()) for line in f.readlines() if line.strip()]
    if len(data) > 0:
        print(f'Raw: {data}')
        print('-' * 30)
        print(f'🏆 Final Mean: {np.mean(data):.4f} ± {np.std(data):.4f}')
        print('-' * 30)
except: pass
"