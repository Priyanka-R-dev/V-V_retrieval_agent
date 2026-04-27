import json
import os
from datetime import datetime

class OpenAICallLogger:
    def __init__(self, log_file="openai_call_log.json"):
        self.log_file = log_file
        # Ensure the log file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as file:
                json.dump([], file)

    def log_call(self, input_tokens, output_tokens):
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * 1.25
        output_cost = (output_tokens / 1_000_000) * 11.0
        total_cost = input_cost + output_cost

        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }

        # Append log entry to the file
        with open(self.log_file, "r+") as file:
            data = json.load(file)
            data.append(log_entry)
            file.seek(0)
            json.dump(data, file, indent=4)

    def get_logs(self):
        # Retrieve all logs
        with open(self.log_file, "r") as file:
            return json.load(file)

    def calculate_grand_total(self):
        # Calculate the grand total of all costs
        with open(self.log_file, "r+") as file:
            data = json.load(file)
            grand_total = sum(entry["total_cost"] for entry in data)
            # Add grand total to the JSON file
            file.seek(0)
            data.append({"grand_total": round(grand_total, 6)})
            file.seek(0)
            json.dump(data, file, indent=4)
        return round(grand_total, 6)

# Example usage
if __name__ == "__main__":
    logger = OpenAICallLogger()
    logger.log_call(input_tokens=5000, output_tokens=2000)
    logs = logger.get_logs()
    print(json.dumps(logs, indent=4))
    grand_total = logger.calculate_grand_total()
    print(f"Grand Total Cost: ${grand_total}")