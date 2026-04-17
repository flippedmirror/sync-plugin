#!/bin/bash
# Monitor remote training progress by polling progress.json
# Usage: ./scripts/monitor_remote_training.sh <IP> <KEY_PATH> <CHECKPOINT_DIR>
#
# Example:
#   ./scripts/monitor_remote_training.sh 3.218.72.176 ~/.keys/sync-plugin-trainer.pem checkpoints/cross_match_v2

IP=${1:?Usage: $0 <IP> <KEY_PATH> <CHECKPOINT_DIR>}
KEY=${2:?Usage: $0 <IP> <KEY_PATH> <CHECKPOINT_DIR>}
CKPT_DIR=${3:-checkpoints/cross_match_v2}

SSH="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i $KEY ec2-user@$IP"

echo "Monitoring training at $IP:$CKPT_DIR"
echo "Press Ctrl+C to stop monitoring"
echo "=========================================="

PREV_EPOCH=0
while true; do
    # Read progress.json
    PROGRESS=$($SSH "cat $CKPT_DIR/progress.json 2>/dev/null" 2>/dev/null)

    if [ -z "$PROGRESS" ]; then
        echo "[$(date +%H:%M:%S)] Waiting for training to start..."
        sleep 10
        continue
    fi

    STATUS=$(echo "$PROGRESS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null)

    if [ "$STATUS" = "complete" ]; then
        echo ""
        echo "=========================================="
        echo "TRAINING COMPLETE!"
        echo "$PROGRESS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Best val_loss: {d.get('best_val_loss', 'N/A')}\")
print(f\"  Total time: {d.get('total_time', 0):.0f}s ({d.get('total_time', 0)/60:.1f} min)\")
"
        echo "=========================================="
        break
    fi

    EPOCH=$(echo "$PROGRESS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('epoch',0))" 2>/dev/null)

    if [ "$EPOCH" != "$PREV_EPOCH" ]; then
        echo "$PROGRESS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
e = d['epoch']
te = d['total_epochs']
pct = e / te * 100
print(f\"[Epoch {e}/{te}] ({pct:.0f}%) {d['phase']} | train={d['train_loss']:.4f} val={d['val_loss']:.4f} | px_mean={d['val_px_mean']:.0f} px_med={d['val_px_median']:.0f} | @20={d['val_hit_20px']:.0%} @50={d['val_hit_50px']:.0%} @100={d['val_hit_100px']:.0%} | act={d['val_action_acc']:.0%} | {d['epoch_time']:.1f}s/epoch | total={d['total_time']:.0f}s\")
" 2>/dev/null
        PREV_EPOCH=$EPOCH
    fi

    sleep 5
done
