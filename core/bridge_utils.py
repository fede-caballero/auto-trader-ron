import os
import subprocess
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class VastAIBridge:
    def __init__(self, instance_id: str, remote_workspace: str = "/workspace"):
        """
        Orchestrates connection to a Vast.ai instance using native SSH/SCP.
        'instance_id' now serves as the SSH Host alias (e.g., 'vast-a10').
        """
        self.ssh_host = instance_id
        self.remote_workspace = remote_workspace

    def sync_push(self, local_path: str, remote_subpath: str = "") -> bool:
        """
        Uploads local file or directory to the Vast.ai instance using SCP.
        """
        remote_dest = os.path.join(self.remote_workspace, remote_subpath).rstrip('/')
        # Ensure remote directory exists
        subprocess.run(["ssh", self.ssh_host, f"mkdir -p {remote_dest}"], capture_output=True)
        
        # Use scp -r for recursive directory copy
        cmd = ["scp", "-r", local_path, f"{self.ssh_host}:{remote_dest}"]
        
        logger.info(f"Pushing {local_path} to Host {self.ssh_host}:{remote_dest} via SCP")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Push successful.")
            return True
        else:
            logger.error(f"Push failed: {result.stderr}")
            return False

    def execute_training(self, script_name: str = "train.py", time_limit_sec: int = 300) -> bool:
        """
        Executes a training script remotely with a strict time limit via SSH.
        """
        # The official Vast.ai PyTorch template stores its environment in /venv/main/
        # We use absolute paths to ensure we hit the correct python process.
        remote_cmd = (
            f"cd {self.remote_workspace} && "
            f"/venv/main/bin/pip install pandas numpy --quiet && "
            f"timeout {time_limit_sec} /venv/main/bin/python3 {script_name}"
        )
        
        cmd = ["ssh", self.ssh_host, remote_cmd]
        
        logger.info(f"Executing remote training with hard limit {time_limit_sec}s: {remote_cmd}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode in [0, 124]:
            if result.returncode == 124:
                logger.warning("Training hit the 5-minute strict timeout limit (expected behavior).")
            else:
                logger.info("Training completed before time limit.")
            return True
        else:
            logger.error(f"Execution failed: {result.stderr}")
            return False

    def sync_pull(self, remote_file: str, local_dest: str) -> bool:
        """
        Downloads a file (e.g., results log) from the Vast.ai instance using SCP.
        """
        remote_src = os.path.join(self.remote_workspace, remote_file)
        cmd = ["scp", f"{self.ssh_host}:{remote_src}", local_dest]
        
        logger.info(f"Pulling {remote_src} to {local_dest} via SCP")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Pull successful.")
            return True
        else:
            logger.error(f"Pull failed: {result.stderr}")
            return False

    def fetch_metric(self, result_log_path: str, metric_key: str = "val_bpb") -> Optional[float]:
        """
        Helper method to parse the pulled results.log
        """
        if not os.path.exists(result_log_path):
            logger.error(f"Log file not found at {result_log_path}")
            return None
        
        # Simplified parsing logic for results.log
        try:
            with open(result_log_path, 'r') as f:
                lines = f.readlines()
            
            for line in reversed(lines):
                if metric_key in line:
                    # Expecting format: "val_bpb: 0.123"
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        return float(parts[1].strip())
            
            logger.warning(f"Metric {metric_key} not found in log.")
            return None
        except Exception as e:
            logger.error(f"Failed to parse metric: {e}")
            return None

# --- Usage Example ---
if __name__ == "__main__":
    # Vast CLI requires API Key config: vastai set api-key <key>
    INSTANCE = "YOUR_INSTANCE_ID"
    
    bridge = VastAIBridge(instance_id=INSTANCE)
    
    # 1. MUTATION: Edit local train.py (done by Anthropic agent externally)
    
    # 2. SYNC PUSH
    if bridge.sync_push("models/train.py"):
        
        # 3. EJECUCIÓN (5 min)
        if bridge.execute_training("train.py", time_limit_sec=300):
            
            # 4. SYNC PULL
            if bridge.sync_pull("results.log", "logs/results.log"):
                
                # 5. REFLEXIÓN
                metric = bridge.fetch_metric("logs/results.log", "val_bpb")
                logger.info(f"Current Training Metric (val_bpb): {metric}")
