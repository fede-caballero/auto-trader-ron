import os
import time
import logging
from dotenv import load_dotenv

from core.bridge_utils import VastAIBridge
from core.mutator import CodeMutator
from core.tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    
    instance_id = os.getenv("VAST_INSTANCE_ID")
    if not instance_id:
        logger.error("VAST_INSTANCE_ID not found in environment. Please set it in .env")
        return

    bridge = VastAIBridge(instance_id=instance_id)
    mutator = CodeMutator(target_file="models/train.py")
    tracker = ExperimentTracker()

    NUM_GENERATIONS = 50
    metric_cache_file = ".best_metric_cache"
    
    # Check if we are resuming a previous session
    if os.path.exists(metric_cache_file):
        with open(metric_cache_file, "r") as f:
            best_val_bpb = float(f.read().strip())
        logger.info(f"Resuming genetic loop from previous best BCE Loss: {best_val_bpb:.6f}")
    else:
        best_val_bpb = float('inf')
        logger.info("Initializing fresh Distributed Karpathy Loop...")

    logger.info("Syncing dataset to remote instance...")
    import subprocess
    subprocess.run(["ssh", os.getenv("VAST_INSTANCE_ID"), "rm -rf /workspace/data"], capture_output=True)
    if not bridge.sync_push("data/", ""):
        logger.warning("Dataset sync returned false, but proceeding anyway.")

    for generation in range(1, NUM_GENERATIONS + 1):
        logger.info(f"=== Starting Generation {generation}/{NUM_GENERATIONS} ===")
        
        # 1. MUTATION
        mutation_details = mutator.mutate()
        
        # 2. SYNC PUSH
        logger.info("Pushing latest code to Vast.ai...")
        # Clean remote directory to prevent scp from nesting it as /workspace/models/models/
        import subprocess
        subprocess.run(["ssh", os.getenv("VAST_INSTANCE_ID"), "rm -rf /workspace/models"], capture_output=True)
        # By pushing 'models/' to the root workspace, scp correctly creates /workspace/models/
        if not bridge.sync_push("models/", ""):
            logger.error("Failed to push code. Aborting generation.")
            tracker.notify_telegram(generation, "Failed - Push Error", None, mutation_details)
            mutator.revert()
            continue
            
        # 3. EJECUCIÓN (15 min per gen)
        logger.info("Executing training script remotely...")
        # Execution time limit is strictly enforced locally via `timeout` 
        # inside `bridge.execute_training()` as well as via `time.time()` inside `train.py`.
        success = bridge.execute_training("models/train.py", time_limit_sec=960)
        
        # 4. SYNC PULL
        logger.info("Pulling results from remote instance...")
        if not bridge.sync_pull("results.log", "logs/results_remote.log"):
            logger.error("Failed to pull logs. Assuming failure.")
            tracker.notify_telegram(generation, "Failed - Pull Error", None, mutation_details)
            mutator.revert()
            continue
            
        # 5. REFLEXIÓN
        val_bpb = bridge.fetch_metric("logs/results_remote.log", "val_bpb")
        
        if val_bpb is not None:
            logger.info(f"Generation {generation} validation metric (val_bpb): {val_bpb:.6f}")
            status = "Success"
            
            if val_bpb < best_val_bpb:
                logger.info(f"NEW BEST! {val_bpb:.6f} < {best_val_bpb:.6f}. Keeping mutation.")
                best_val_bpb = val_bpb
                status = "Success - New Best"
                
                # Save cache for seamless resume
                with open(metric_cache_file, "w") as f:
                    f.write(str(best_val_bpb))
                
                # Download new champion weights
                logger.info("Downloading champion model weights (.pth)...")
                bridge.sync_pull("best_model.pth", "models/champion_model.pth")
            else:
                logger.info(f"Did not improve ({val_bpb:.6f} >= {best_val_bpb:.6f}). Reverting mutation.")
                mutator.revert()
                status = "Success - Reverted"
        else:
            logger.warning(f"Generation {generation} failed to produce a valid metric. Reverting.")
            mutator.revert()
            status = "Failed - Metric Not Found"
            
        # Notify via webhook (n8n -> Telegram)
        tracker.notify_telegram(generation, status, val_bpb, mutation_details)
        
        logger.info(f"=== Generation {generation} Complete ===\n")
        
        # Small delay between runs to let Vast AI connections cool down if needed
        time.sleep(2)

    logger.info("Auto-Evolution process fully completed.")

if __name__ == "__main__":
    main()
