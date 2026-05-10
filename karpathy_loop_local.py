#!/usr/bin/env python3
"""
Local Karpathy Loop — CPU Training (No Vast.ai Required)
Trains the HybridLSTMTransformer model locally using genetic mutation.
Designed for machines with >=8GB RAM and no GPU.

Usage:
    python3 karpathy_loop_local.py                  # Train locally
    python3 karpathy_loop_local.py --deploy aws-bot # Train + SCP champion to AWS
"""
import os
import sys
import csv
import time
import json
import logging
import subprocess
import shutil
import statistics
from datetime import datetime, timezone

from dotenv import load_dotenv
from core.mutator import CodeMutator
from core.prepare import DataPreparer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ──────────────── CONFIGURATION ────────────────
SYMBOL = "RONINUSDT"
INTERVAL = "4h"
NUM_GENERATIONS = 50
METRIC_CACHE_FILE = ".best_fitness_cache"  # higher = better (PnL-based fitness)
METRIC_NAME = "val_fitness"
SEEDS = [42, 1337, 2024]            # one training run per seed; mean drives the decision
HISTORY_CSV = "logs/karpathy_history.csv"
# ────────────────────────────────────────────────

def download_fresh_data():
    """Download the latest market data from Binance for training."""
    preparer = DataPreparer()
    
    logger.info(f"Downloading 1000 klines for {SYMBOL} at {INTERVAL}...")
    raw_file = preparer.download_binance_klines(SYMBOL, INTERVAL, limit=1000)
    if not raw_file:
        logger.error("Failed to download data from Binance!")
        return False
    
    logger.info("Creating features from raw data...")
    processed_file = preparer.create_features(raw_file)
    if not processed_file:
        logger.error("Failed to process features!")
        return False
    
    logger.info(f"Data ready: {processed_file}")
    return True

def run_training_locally(seed: int):
    """Execute models/train.py locally on CPU with a pinned seed."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
    env["TRAIN_SEED"] = str(seed)

    result = subprocess.run(
        [sys.executable, "models/train.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True, timeout=1800,  # 30 min max
        env=env
    )

    if result.returncode != 0:
        logger.error(f"Training failed (seed={seed}):\n{result.stderr[-500:]}")
        return False

    logger.info(f"Training completed (seed={seed}).")
    return True

def fetch_metric(log_file, metric_name="val_bpb"):
    """Parse the results log to extract the validation metric."""
    if not os.path.exists(log_file):
        return None
    with open(log_file, "r") as f:
        for line in f:
            if metric_name in line:
                try:
                    return float(line.split(":")[-1].strip())
                except ValueError:
                    pass
    return None

def init_history_csv():
    """Create the per-generation history CSV if it doesn't exist."""
    os.makedirs(os.path.dirname(HISTORY_CSV), exist_ok=True)
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "generation", "mutation_key", "mutation_value",
                "seed_metrics", "fitness_mean", "fitness_std", "fitness_max",
                "decision", "best_so_far"
            ])

def log_generation(generation, mutation, seed_metrics, decision, best_so_far):
    """Append a single row describing this generation's outcome."""
    valid = [m for m in seed_metrics if m is not None]
    mean = statistics.mean(valid) if valid else None
    stdev = statistics.stdev(valid) if len(valid) >= 2 else 0.0
    maximum = max(valid) if valid else None

    mutation_key = next(iter(mutation.keys()), "") if mutation else ""
    mutation_value = mutation.get(mutation_key, "") if mutation else ""

    with open(HISTORY_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(),
            generation, mutation_key, mutation_value,
            json.dumps(seed_metrics),
            f"{mean:.6f}" if mean is not None else "",
            f"{stdev:.6f}",
            f"{maximum:.6f}" if maximum is not None else "",
            decision,
            f"{best_so_far:.6f}" if best_so_far != float('-inf') else "",
        ])

def main():
    load_dotenv()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    deploy_host = None
    if "--deploy" in sys.argv:
        idx = sys.argv.index("--deploy")
        if idx + 1 < len(sys.argv):
            deploy_host = sys.argv[idx + 1]
            logger.info(f"Will deploy champion to: {deploy_host}")
    
    mutator = CodeMutator(target_file="models/train.py")
    init_history_csv()

    # Load or initialize best fitness (HIGHER is better — PnL-based)
    if os.path.exists(METRIC_CACHE_FILE):
        with open(METRIC_CACHE_FILE, "r") as f:
            best_fitness = float(f.read().strip())
        logger.info(f"Resuming from previous best fitness: {best_fitness:+.4f}")
    else:
        best_fitness = float('-inf')
        logger.info("Initializing fresh Local Karpathy Loop (fitness-based)...")

    # Download fresh market data
    if not download_fresh_data():
        logger.error("Aborting: Could not download market data.")
        return

    results_log = "results.log"

    for generation in range(1, NUM_GENERATIONS + 1):
        logger.info(f"=== Generation {generation}/{NUM_GENERATIONS} ===")

        # 1. MUTATION
        mutation_details = mutator.mutate()

        # 2. Train NUM_REPLICAS times with different seeds; track best individual replica
        seed_metrics = []
        best_replica_fitness = float('-inf')
        best_replica_files = (None, None)  # (model_path, meta_path) of best individual run

        gen_start = time.time()
        for seed in SEEDS:
            # Clean per-replica artifacts
            for f in (results_log, "best_model.pth", "champion_meta.json"):
                if os.path.exists(f):
                    os.remove(f)

            logger.info(f"  ↳ Replica seed={seed}...")
            r_start = time.time()
            success = run_training_locally(seed=seed)
            r_elapsed = time.time() - r_start

            if not success:
                seed_metrics.append(None)
                continue

            metric = fetch_metric(results_log, METRIC_NAME)
            seed_metrics.append(metric)
            if metric is None:
                logger.warning(f"  ↳ seed={seed}: no {METRIC_NAME} found ({r_elapsed:.1f}s)")
                continue
            # Pull supporting metrics for the log line
            pnl_pct = fetch_metric(results_log, "val_pnl_pct")
            trades = fetch_metric(results_log, "val_trades")
            logger.info(
                f"  ↳ seed={seed}: fitness={metric:+.4f} "
                f"(pnl={pnl_pct:+.2f}% trades={int(trades) if trades is not None else '?'}) "
                f"({r_elapsed:.1f}s)"
            )

            # Stash the best individual replica's artifacts for later promotion
            if metric > best_replica_fitness and os.path.exists("best_model.pth"):
                best_replica_fitness = metric
                shutil.copy2("best_model.pth", "best_model_gen.pth")
                if os.path.exists("champion_meta.json"):
                    shutil.copy2("champion_meta.json", "champion_meta_gen.json")
                    best_replica_files = ("best_model_gen.pth", "champion_meta_gen.json")
                else:
                    best_replica_files = ("best_model_gen.pth", None)

        gen_elapsed = time.time() - gen_start
        valid = [m for m in seed_metrics if m is not None]
        logger.info(f"Generation {generation} took {gen_elapsed:.1f}s | seed_metrics={seed_metrics}")

        # 3. DECIDE on the MEAN of valid replicas (HIGHER fitness = better)
        if not valid:
            logger.warning("All replicas failed. Reverting mutation.")
            mutator.revert()
            log_generation(generation, mutation_details, seed_metrics, "reverted_failed", best_fitness)
        else:
            mean_fit = statistics.mean(valid)
            stdev_fit = statistics.stdev(valid) if len(valid) >= 2 else 0.0
            logger.info(f"Generation {generation} mean(fitness)={mean_fit:+.4f}  std={stdev_fit:.4f}  (n={len(valid)})")

            if mean_fit > best_fitness:
                logger.info(f"🏆 NEW BEST! mean {mean_fit:+.4f} > {best_fitness:+.4f}. Keeping mutation.")
                best_fitness = mean_fit

                with open(METRIC_CACHE_FILE, "w") as f:
                    f.write(str(best_fitness))

                # Promote the best individual replica's artifacts as champion
                model_src, meta_src = best_replica_files
                if model_src and os.path.exists(model_src):
                    shutil.copy2(model_src, "models/champion_model.pth")
                    logger.info(f"Champion model promoted (best replica fitness={best_replica_fitness:+.4f})")
                if meta_src and os.path.exists(meta_src):
                    shutil.copy2(meta_src, "models/champion_meta.json")
                    logger.info("Champion metadata saved to models/champion_meta.json")
                else:
                    logger.warning("champion_meta.json not produced — inference will fall back to legacy stats.")

                log_generation(generation, mutation_details, seed_metrics, "kept", best_fitness)
            else:
                logger.info(f"Did not improve (mean {mean_fit:+.4f} <= {best_fitness:+.4f}). Reverting.")
                mutator.revert()
                log_generation(generation, mutation_details, seed_metrics, "reverted", best_fitness)

        # Cleanup per-generation stash files
        for f in ("best_model_gen.pth", "champion_meta_gen.json"):
            if os.path.exists(f):
                os.remove(f)

        logger.info(f"=== Generation {generation} Complete ===\n")
        time.sleep(1)
    
    logger.info(f"Evolution complete! Best fitness: {best_fitness:+.4f}")
    
    # Auto-deploy if requested
    if deploy_host and os.path.exists("models/champion_model.pth"):
        logger.info(f"Deploying champion to {deploy_host}...")
        bot_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        files_to_deploy = ["models/champion_model.pth"]
        if os.path.exists("models/champion_meta.json"):
            files_to_deploy.append("models/champion_meta.json")
        else:
            logger.warning("No champion_meta.json to deploy — VPS predict.py will use legacy fallback.")
        result = subprocess.run(
            ["scp", *files_to_deploy, f"{deploy_host}:~/{bot_name}/models/"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"✅ Champion ({len(files_to_deploy)} file(s)) deployed to {deploy_host} successfully!")
        else:
            logger.error(f"Deploy failed: {result.stderr}")

if __name__ == "__main__":
    main()
