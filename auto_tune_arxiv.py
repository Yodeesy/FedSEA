import subprocess
import itertools
import time
import re
import os
import sys

# =================配置区域=================
BASE_CMD = (
    "python -u main.py "  # -u 禁用缓冲，强制实时输出
    "--dataset ogbn-arxiv "
    "--model GCN "
    "--num_layers 3 "
    "--hidden_dim 256 "
    "--num_rounds 100 "  # ⚠️ 改成100轮！调参没必要跑200，太慢了
    "--patience 20 "  # ⚠️ 降低耐心值，快速止损
    "--server_lr 0.01 "
    "--gen_train_steps 50 "
    "--gen_num_samples 1 "
    "--device_id 3 "  # GPU 3
    "--seed 42 "
)

search_space = {
    "w_proto": [10.0, 20.0],  # 减少搜索空间，先跑核心的
    "gen_knn": [10, 20],
    "gen_lr": [0.001, 0.0005],
    "dropout": [0.5],
    "w_ot": [0.01]
}

LOG_FILE = "arxiv_tuning_log.csv"


# =========================================

def parse_result(output):
    acc_online = 0.0
    acc_ema = 0.0
    match_online = re.search(r"Final Best Result \(Online\):\s+([\d\.]+)", output)
    if match_online: acc_online = float(match_online.group(1))
    match_ema = re.search(r"Final Best Result \(EMA\):\s+([\d\.]+)", output)
    if match_ema: acc_ema = float(match_ema.group(1))
    return acc_online, acc_ema


def run_tuning():
    # 初始化 CSV (带 flush)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("w_proto,gen_knn,gen_lr,w_ot,dropout,Online_Acc,EMA_Acc,Best_Hybrid\n")
            f.flush()  # 强制写入硬盘

    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_jobs = len(combinations)
    print(f"🚀 [Auto-Tuner] Starting Grid Search for Arxiv on GPU 3")
    print(f"📋 Total Configurations: {total_jobs}")
    print("=" * 60)

    for idx, config in enumerate(combinations):
        print(f"\n▶️  [{idx + 1}/{total_jobs}] Running: {config}")

        param_str = " ".join([f"--{k} {v}" for k, v in config.items()])
        full_cmd = f"{BASE_CMD} {param_str}"

        start_time = time.time()

        # ✅ 改进：使用 Popen 实时打印子进程输出
        output_buffer = ""
        try:
            process = subprocess.Popen(
                full_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # 行缓冲
                universal_newlines=True
            )

            # 实时读取输出，这样你就知道它卡在哪了
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.strip())  # 打印到屏幕
                    output_buffer += line  # 存起来用于解析

            # 解析
            online, ema = parse_result(output_buffer)
            best_hybrid = max(online, ema)
            duration = time.time() - start_time

            status = f"✅ Done ({duration:.1f}s) | Hybrid: {best_hybrid:.4f}" if best_hybrid > 0 else "⚠️ Failed"
            print(status)

            # 写入 CSV 并立即 Flush
            with open(LOG_FILE, "a") as f:
                f.write(
                    f"{config['w_proto']},{config['gen_knn']},{config['gen_lr']},{config['w_ot']},{config['dropout']},{online},{ema},{best_hybrid}\n")
                f.flush()  # ✅ 关键：强制保存，防止断电白跑

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_tuning()