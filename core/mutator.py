import os
import re
import random
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CodeMutator:
    def __init__(self, target_file: str = "models/train.py"):
        self.target_file = target_file
        self.hyperparameters = {
            "LR": {"type": "float", "range": [1e-4, 5e-3]},
            "BATCH_SIZE": {"type": "choice", "options": [32, 64, 128]},
            "SEQ_LEN": {"type": "int", "range": [20, 60]}
        }
        self.last_mutation = {}
        self.backup_content = None

    def backup(self):
        """Saves the current state of the file before mutating"""
        if os.path.exists(self.target_file):
            with open(self.target_file, 'r') as f:
                self.backup_content = f.read()

    def revert(self):
        """Reverts to the backup if the mutation performed worse"""
        if self.backup_content is not None:
            with open(self.target_file, 'w') as f:
                f.write(self.backup_content)
            logger.info("Reverted to previous best code state.")

    def mutate(self) -> Dict[str, Any]:
        """
        Randomly mutates hyperparameters in the target script as a proof of concept
        for the auto-evolutionary step.
        """
        self.backup()
        
        if not self.backup_content:
            logger.error(f"Cannot mutate, file {self.target_file} not found or empty.")
            return {}

        content = self.backup_content
        mutations_applied = {}

        # We will pick one random hyperparameter to mutate per step
        hp_key = random.choice(list(self.hyperparameters.keys()))
        hp_config = self.hyperparameters[hp_key]

        new_val = None
        if hp_config["type"] == "float":
            new_val = random.uniform(hp_config["range"][0], hp_config["range"][1])
            new_val_str = f"{new_val:.5f}"
        elif hp_config["type"] == "int":
            new_val = random.randint(hp_config["range"][0], hp_config["range"][1])
            new_val_str = str(new_val)
        elif hp_config["type"] == "choice":
            new_val = random.choice(hp_config["options"])
            new_val_str = str(new_val)

        # Regex replacement in the python script. 
        # Looking for things like: LR = 1e-3 or LR=0.001
        pattern = rf"^({hp_key}\s*=\s*)[^\n]+"
        
        # Check if the pattern exists in the file (using multiline match)
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, rf"\g<1>{new_val_str}", content, flags=re.MULTILINE)
            mutations_applied[hp_key] = new_val
            logger.info(f"Mutated {hp_key} -> {new_val_str}")
            self.last_mutation = mutations_applied
            
            with open(self.target_file, 'w') as f:
                f.write(content)
        else:
            logger.warning(f"Hyperparameter {hp_key} not found in {self.target_file} using regex.")

        return mutations_applied
