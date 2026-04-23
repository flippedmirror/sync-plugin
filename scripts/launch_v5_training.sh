#!/bin/bash
# End-to-end v5 training pipeline for L40S GPU instances.
#
# Incorporates all learnings:
#   - Python 3.12 (not 3.9) for torch compatibility
#   - cache-batch-size=16 to avoid cublasLtCreate crash on CUDA 13.0
#   - rsvg-convert for SVG icon rendering (not cairosvg)
#   - -u flag for unbuffered Python output (visible in nohup logs)
#   - Icon packs downloaded from GitHub
#   - Feature caching enabled (4-5x faster epochs)
#   - progress.json for remote monitoring
#
# Usage:
#   # Full pipeline: datagen + training
#   ./scripts/launch_v5_training.sh <IP> <KEY_PATH> --pairs 5000 --epochs 5
#
#   # Training only (data already on instance or uploaded via S3)
#   ./scripts/launch_v5_training.sh <IP> <KEY_PATH> --data-dir data/cross_match_v5_5k --epochs 5
#
#   # Full run with fine-tuning
#   ./scripts/launch_v5_training.sh <IP> <KEY_PATH> --pairs 20000 --epochs 30 --finetune-epochs 5
#
#   # Resume Phase 2 from checkpoint
#   ./scripts/launch_v5_training.sh <IP> <KEY_PATH> --data-dir data/cross_match_v5 --epochs 5 \
#       --finetune-epochs 5 --resume checkpoints/v5/best.pt --lr 1e-6

set -euo pipefail

# ─── Parse Args ───

IP=""
KEY=""
PAIRS=0
EPOCHS=5
FINETUNE_EPOCHS=0
BATCH_SIZE=64
LR="1e-4"
WARMUP=2
DATA_DIR=""
OUTPUT_DIR=""
RESUME=""
SEED=42
S3_DATA=""
SKIP_SETUP=false

usage() {
    echo "Usage: $0 <IP> <KEY_PATH> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --pairs N          Generate N synthetic pairs (skip if --data-dir set)"
    echo "  --data-dir PATH    Use existing data dir on instance (skip datagen)"
    echo "  --s3-data S3_URI   Download data from S3 (e.g. s3://bucket/data.tar.gz)"
    echo "  --output-dir PATH  Checkpoint output dir (default: checkpoints/v5_<pairs>_<epochs>ep)"
    echo "  --epochs N         Number of training epochs (default: 5)"
    echo "  --finetune-epochs N  Fine-tune epochs at end (default: 0)"
    echo "  --batch-size N     Training batch size (default: 64)"
    echo "  --lr RATE          Learning rate (default: 1e-4)"
    echo "  --warmup N         Warmup epochs (default: 2)"
    echo "  --resume PATH      Resume from checkpoint on instance"
    echo "  --seed N           Random seed for datagen (default: 42)"
    echo "  --skip-setup       Skip dependency installation (instance already set up)"
    exit 1
}

# First two positional args are IP and KEY
[ $# -lt 2 ] && usage
IP="$1"; shift
KEY="$1"; shift

while [ $# -gt 0 ]; do
    case "$1" in
        --pairs) PAIRS="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --s3-data) S3_DATA="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --finetune-epochs) FINETUNE_EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP=true; shift ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

# Defaults
if [ -z "$DATA_DIR" ] && [ "$PAIRS" -eq 0 ] && [ -z "$S3_DATA" ]; then
    echo "Error: specify --pairs, --data-dir, or --s3-data"
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    DATA_DIR="data/cross_match_v5_${PAIRS}p"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="checkpoints/v5_${PAIRS}p_${EPOCHS}ep"
fi

SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i $KEY ec2-user@$IP"
SCP="scp -o StrictHostKeyChecking=no -i $KEY"

echo "============================================"
echo "  CrossMatch V5 Training Pipeline"
echo "============================================"
echo "  Instance:     $IP"
echo "  Data:         ${DATA_DIR} (${PAIRS} pairs to generate)"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Epochs:       ${EPOCHS} + ${FINETUNE_EPOCHS} fine-tune"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  LR:           ${LR}"
echo "  Resume:       ${RESUME:-none}"
echo "============================================"
echo ""

# ─── Step 1: Instance Setup ───

if [ "$SKIP_SETUP" = false ]; then
    echo "[1/6] Checking GPU..."
    $SSH "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"

    echo "[2/6] Installing dependencies..."
    $SSH "pip3.12 install --quiet torch torchvision pillow 2>&1 | tail -2"
    $SSH "python3.12 -c 'import torch; assert torch.cuda.is_available(); print(\"CUDA OK:\", torch.cuda.get_device_name(0))'"

    echo "[3/6] Setting up repo and icons..."
    $SSH "if [ ! -d ~/sync-plugin ]; then git clone --quiet https://github.com/flippedmirror/sync-plugin.git; else cd ~/sync-plugin && git pull --quiet; fi"
    $SSH "cd ~/sync-plugin && if [ ! -d data/icons/phosphor/core-main ]; then
        mkdir -p data/icons/phosphor data/icons/simple-icons
        cd data/icons/phosphor && curl -sL https://github.com/phosphor-icons/core/archive/refs/heads/main.zip -o p.zip && unzip -q p.zip
        cd ../simple-icons && curl -sL https://github.com/simple-icons/simple-icons/archive/refs/heads/develop.zip -o s.zip && unzip -q s.zip
        echo 'Icons downloaded'
    else
        echo 'Icons already present'
    fi"
else
    echo "[1-3/6] Skipping setup (--skip-setup)"
    $SSH "cd ~/sync-plugin && git pull --quiet"
fi

# ─── Step 2: Data ───

if [ -n "$S3_DATA" ]; then
    echo "[4/6] Downloading data from S3..."
    $SSH "cd ~/sync-plugin && aws s3 cp $S3_DATA /tmp/data.tar.gz && tar xzf /tmp/data.tar.gz -C . && rm /tmp/data.tar.gz"
elif [ "$PAIRS" -gt 0 ]; then
    echo "[4/6] Checking if data exists..."
    DATA_EXISTS=$($SSH "[ -f ~/sync-plugin/${DATA_DIR}/annotations.json ] && echo yes || echo no" 2>/dev/null)
    if [ "$DATA_EXISTS" = "yes" ]; then
        echo "  Data already exists at ${DATA_DIR}, skipping generation."
    else
        echo "[4/6] Generating ${PAIRS} synthetic pairs (this takes ~${PAIRS}/110 minutes)..."
        $SSH "cd ~/sync-plugin && python3.12 -u -m cross_match.synthetic_v5 --output-dir ${DATA_DIR} --num-pairs ${PAIRS} --seed ${SEED}" &
        DATAGEN_PID=$!
        # Monitor datagen
        while kill -0 $DATAGEN_PID 2>/dev/null; do
            COUNT=$($SSH "ls ~/sync-plugin/${DATA_DIR}/source/ 2>/dev/null | wc -l" 2>/dev/null || echo "?")
            echo "  [$(date +%H:%M:%S)] Generated ${COUNT}/${PAIRS} pairs..."
            sleep 30
        done
        wait $DATAGEN_PID
        echo "  Data generation complete."
    fi
else
    echo "[4/6] Using existing data at ${DATA_DIR}"
fi

# ─── Step 3: Build Training Command ───

TRAIN_CMD="python3.12 -u -m cross_match.train"
TRAIN_CMD="${TRAIN_CMD} --data-dir ${DATA_DIR}"
TRAIN_CMD="${TRAIN_CMD} --output-dir ${OUTPUT_DIR}"
TRAIN_CMD="${TRAIN_CMD} --epochs ${EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --finetune-epochs ${FINETUNE_EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --batch-size ${BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --lr ${LR}"
TRAIN_CMD="${TRAIN_CMD} --warmup-epochs ${WARMUP}"
TRAIN_CMD="${TRAIN_CMD} --cache-batch-size 16"
TRAIN_CMD="${TRAIN_CMD} --device cuda"

if [ -n "$RESUME" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume ${RESUME}"
fi

# ─── Step 4: Launch Training ───

echo "[5/6] Launching training..."
echo "  Command: ${TRAIN_CMD}"
$SSH "cd ~/sync-plugin && nohup ${TRAIN_CMD} > training_v5.log 2>&1 & echo \$!"
TRAIN_PID=$($SSH "pgrep -f 'cross_match.train' | head -1" 2>/dev/null)
echo "  Training PID: ${TRAIN_PID}"

# ─── Step 5: Monitor ───

echo "[6/6] Monitoring..."
echo ""

PREV_EPOCH=0
while true; do
    # Check if process is still running
    RUNNING=$($SSH "pgrep -f 'cross_match.train' | wc -l" 2>/dev/null || echo "0")
    if [ "$RUNNING" = "0" ]; then
        # Check if it completed or crashed
        PROGRESS=$($SSH "cat ~/sync-plugin/${OUTPUT_DIR}/progress.json 2>/dev/null" 2>/dev/null || echo "")
        if echo "$PROGRESS" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status')=='complete' else 1)" 2>/dev/null; then
            echo ""
            echo "============================================"
            echo "  TRAINING COMPLETE"
            echo "============================================"
            echo "$PROGRESS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Best val_loss: {d.get('best_val_loss', 'N/A'):.4f}\")
print(f\"  Total time: {d.get('total_time', 0):.0f}s ({d.get('total_time', 0)/60:.1f} min)\")
"
            break
        else
            echo ""
            echo "!!! Training process died. Last log output:"
            $SSH "tail -10 ~/sync-plugin/training_v5.log" 2>/dev/null
            exit 1
        fi
    fi

    # Read progress
    PROGRESS=$($SSH "cat ~/sync-plugin/${OUTPUT_DIR}/progress.json 2>/dev/null" 2>/dev/null || echo "")
    if [ -z "$PROGRESS" ]; then
        # Check raw log for caching status
        LAST_LINE=$($SSH "tail -1 ~/sync-plugin/training_v5.log 2>/dev/null" 2>/dev/null || echo "")
        echo "  [$(date +%H:%M:%S)] ${LAST_LINE:-Initializing...}"
        sleep 10
        continue
    fi

    EPOCH=$(echo "$PROGRESS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('epoch',0))" 2>/dev/null || echo "0")
    if [ "$EPOCH" != "$PREV_EPOCH" ]; then
        echo "$PROGRESS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
e, te = d['epoch'], d['total_epochs']
print(f\"  [Epoch {e}/{te}] {d['phase']} | train={d['train_loss']:.4f} val={d['val_loss']:.4f} | px_mean={d['val_px_mean']:.0f} px_med={d['val_px_median']:.0f} | @20={d['val_hit_20px']:.0%} @50={d['val_hit_50px']:.0%} @100={d['val_hit_100px']:.0%} | act={d['val_action_acc']:.0%} | {d['epoch_time']:.1f}s\")
" 2>/dev/null
        PREV_EPOCH=$EPOCH
    fi
    sleep 10
done

echo ""
echo "Download checkpoint:"
echo "  $SCP ec2-user@$IP:~/sync-plugin/${OUTPUT_DIR}/best.pt checkpoints/"
echo ""
echo "View full log:"
echo "  $SSH 'cat ~/sync-plugin/training_v5.log'"
