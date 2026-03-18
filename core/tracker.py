import os
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExperimentTracker:
    def __init__(self):
        self.n8n_webhook_url = os.getenv("N8N_WEBHOOK_URL")
        if not self.n8n_webhook_url:
            logger.warning("N8N_WEBHOOK_URL is not set in environment. Notifications will be disabled.")

    def notify_telegram(self, experiment_id: int, status: str, metric: Optional[float], mutation: Dict[str, Any]) -> bool:
        """
        Sends an update to the n8n webhook which routes to Telegram.
        """
        if not self.n8n_webhook_url:
            return False
            
        payload = {
            "experiment_id": experiment_id,
            "status": status,
            "metric": metric if metric is not None else "N/A",
            "mutation_details": str(mutation)
        }
        
        try:
            logger.info(f"Sending tracking info to n8n webhook for EXP-{experiment_id}")
            response = requests.post(self.n8n_webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.debug("Successfully notified n8n.")
                return True
            else:
                logger.warning(f"Webhook returned status code {response.status_code}: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send webhook: {e}")
            return False
