#!/bin/bash
# Launch training on a remote GPU instance (end-to-end).
# Usage: ./scripts/launch_remote_training.sh <IP> <KEY_PATH>
#
# Example:
#   ./scripts/launch_remote_training.sh 3.218.72.176 ~/.keys/sync-plugin-trainer.pem

set -e

IP=${1:?Usage: $0 <IP> <KEY_PATH>}
KEY=${2:?Usage: $0 <IP> <KEY_PATH>}

SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i $KEY ec2-user@$IP"
SCP="scp -o StrictHostKeyChecking=no -i $KEY"

echo "=== Remote Training Launcher ==="
echo "Instance: $IP"
echo ""

# Step 1: Check GPU
echo "[1/5] Checking GPU..."
$SSH "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

# Step 2: Install deps
echo "[2/5] Installing Python dependencies..."
$SSH "pip3 install --quiet torch torchvision transformers Pillow 2>&1 | tail -1"
$SSH "python3 -c 'import torch; assert torch.cuda.is_available(), \"CUDA not available!\"; print(\"CUDA OK:\", torch.cuda.get_device_name(0))'"

# Step 3: Push code
echo "[3/5] Pushing code..."
$SCP -r cross_match ec2-user@$IP:~/
echo "  Code synced."

# Step 4: Generate synthetic data (if not already present)
echo "[4/5] Generating 10K synthetic training pairs..."
$SSH "python3 -c 'import os; print(\"exists\" if os.path.exists(\"data/cross_match_v2/annotations.json\") else \"missing\")'" > /tmp/data_check.txt 2>/dev/null
if grep -q "missing" /tmp/data_check.txt; then
    $SSH "python3 -u -m cross_match.synthetic_v2 --output-dir data/cross_match_v2 --num-pairs 10000 2>&1 | tail -3"
else
    echo "  Data already exists, skipping."
fi

# Step 5: Launch training
echo "[5/5] Launching training (50 epochs, nohup)..."
$SSH "nohup python3 -u -m cross_match.train --data-dir data/cross_match_v2 --output-dir checkpoints/cross_match_v2 --device cuda --batch-size 32 --epochs 50 --finetune-epochs 5 > training_v2.log 2>&1 & echo PID=\$!"

echo ""
echo "=== Training launched! ==="
echo ""
echo "Monitor progress:"
echo "  ./scripts/monitor_remote_training.sh $IP $KEY checkpoints/cross_match_v2"
echo ""
echo "View raw log:"
echo "  ssh -i $KEY ec2-user@$IP 'tail -20 training_v2.log'"
echo ""
echo "Download best checkpoint when done:"
echo "  scp -i $KEY ec2-user@$IP:~/checkpoints/cross_match_v2/best.pt checkpoints/"
