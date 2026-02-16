import csv
import os
from datetime import datetime

class ContextMetricsLogger:
    def __init__(self, filepath="context_metrics.csv"):
        self.filepath = filepath
        self.turn = 0
        
        self.fields = [
            "timestamp",
            "turn",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "token_budget",
            "summarization_triggered",
            "l1_tokens",
            "l2_tokens",
            "raw_buffer_len",
            "latency_seconds",
            "tokens_per_second"
        ]

        # Create file with header if not exists
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log(
        self,
        input_tokens,
        output_tokens,
        token_budget,
        summarization_triggered,
        l1_tokens,
        l2_tokens,
        raw_buffer_len,
        latency_seconds
    ):
        self.turn += 1
        
        total_tokens = input_tokens + output_tokens
        tokens_per_second = (
            output_tokens / latency_seconds if latency_seconds > 0 else 0
        )

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turn": self.turn,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "token_budget": token_budget,
            "summarization_triggered": int(summarization_triggered),
            "l1_tokens": l1_tokens,
            "l2_tokens": l2_tokens,
            "raw_buffer_len": raw_buffer_len,
            "latency_seconds": round(latency_seconds, 4),
            "tokens_per_second": round(tokens_per_second, 2)
        }

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)
