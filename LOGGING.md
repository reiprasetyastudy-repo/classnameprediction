# Logging System

All scripts now include comprehensive logging to track execution history and enable monitoring of long-running processes.

## Log Files Location

All logs are stored in the `logs/` directory with timestamps:

```
logs/
├── eval_gpu_20250127_143022.log
├── eval_gpu_20250127_150145.log
├── train_20250127_091530.log
└── build_dataset_20250127_083012.log
```

## Log File Format

Each log entry includes:
- **Timestamp**: When the event occurred
- **Log Level**: INFO, WARNING, ERROR, etc.
- **Message**: Detailed information about the operation

Example log entry:
```
2025-01-27 14:30:22 - eval_gpu - INFO - Using GPU: NVIDIA GeForce RTX 5090
2025-01-27 14:30:22 - eval_gpu - INFO - Batch size: 16
2025-01-27 14:30:23 - eval_gpu - INFO - Model and tokenizer loaded
2025-01-27 14:30:25 - eval_gpu - INFO - Loaded 3256 test examples
```

## Monitoring Long-Running Processes

### Real-Time Monitoring with `tail -f`

When you run a script, it will print the log file location:

```bash
python scripts/eval_gpu.py --ckpt model/checkpoints/run1-java --data datasets/java --k 5
```

Output:
```
2025-01-27 14:30:22 - INFO - Logging to: logs/eval_gpu_20250127_143022.log
2025-01-27 14:30:22 - INFO - Monitor progress: tail -f logs/eval_gpu_20250127_143022.log
```

### Monitor in Real-Time

Open a new terminal and run:

```bash
tail -f logs/eval_gpu_20250127_143022.log
```

This will show:
- ✅ Real-time progress updates
- ✅ Configuration parameters
- ✅ Current status
- ✅ Errors or warnings

### Running in Background

Run script in background and monitor separately:

```bash
# Start in background
nohup python scripts/eval_gpu.py --ckpt model/checkpoints/run1-java --data datasets/java --k 5 > /dev/null 2>&1 &

# Get the log file name (it will be the newest in logs/)
LOG_FILE=$(ls -t logs/eval_gpu_*.log | head -1)

# Monitor progress
tail -f $LOG_FILE
```

Or use a simpler approach:

```bash
# Run in background, redirect output to log
python scripts/eval_gpu.py --ckpt model/checkpoints/run1-java --data datasets/java --k 5 &

# Monitor the latest log file
tail -f $(ls -t logs/eval_gpu_*.log | head -1)
```

## Recovering from Disconnection

If you get disconnected from a remote server:

1. **Reconnect to server**
2. **Find the log file**:
   ```bash
   # List recent logs
   ls -lt logs/

   # Find logs from today
   ls -l logs/*_$(date +%Y%m%d)_*.log
   ```

3. **Check progress**:
   ```bash
   # See last 50 lines
   tail -50 logs/eval_gpu_20250127_143022.log

   # Continue monitoring
   tail -f logs/eval_gpu_20250127_143022.log
   ```

4. **Check if process is still running**:
   ```bash
   ps aux | grep eval_gpu
   ```

## Log Information by Script

### eval_gpu.py / eval.py
Logs include:
- Device information (GPU/CPU)
- Model and data paths
- Batch size and configuration
- Number of test examples
- Evaluation progress
- Final metrics (EM, Top-K, Levenshtein)
- Elapsed time and speed

### train.py / train_codegen.py
Logs include:
- Model initialization
- Training configuration (batch size, learning rate, epochs)
- Device information
- Dataset sizes
- Training progress (loss, metrics)
- Checkpoint saves
- Total training time

### build_dataset.py
Logs include:
- Repository URLs being cloned
- Clone success/failure status
- Number of classes extracted per repo
- Dataset split sizes
- Total examples generated

## Tips

### Find Specific Information

```bash
# Find all evaluation runs
ls logs/eval_*.log

# Search for errors
grep -i error logs/eval_gpu_20250127_143022.log

# Find training start times
grep "Training" logs/train_*.log

# See all GPU used
grep "Using GPU" logs/*.log

# Check completion status
grep "completed" logs/*.log
```

### Archive Old Logs

```bash
# Archive logs older than 30 days
find logs/ -name "*.log" -mtime +30 -exec gzip {} \;

# Or move to archive folder
mkdir -p logs/archive
find logs/ -name "*.log" -mtime +30 -exec mv {} logs/archive/ \;
```

### Filter by Date

```bash
# Today's logs
ls logs/*_$(date +%Y%m%d)_*.log

# Yesterday's logs
ls logs/*_$(date -d yesterday +%Y%m%d)_*.log

# Specific date
ls logs/*_20250127_*.log
```

## Console Output

**Important**: All original console output (print statements) remains unchanged. Logging is **additive** - you get:

1. ✅ **Console output** - Same as before (for interactive use)
2. ✅ **Log files** - Persistent history (for monitoring and recovery)

Both contain the same information, but log files are persistent and can be monitored with `tail -f`.
