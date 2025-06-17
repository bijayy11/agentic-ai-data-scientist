import sys, os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents_config import objective_agent

def main():
    # Simulate a conversation history as Autogen would provide
    messages = [
        {
            "role": "user",
            "content": json.dumps({
                "function": "generate_objective",
                "arguments": {
                    "user_prompt": "Build a predictive model to estimate loan default risk based on applicant data"
                }
            })
        }
    ]

    response = objective_agent._generate_response(messages)
    print("\n=== Agent Response ===")
    print(response)  # Will be a JSON string with the result from generate_objective

if __name__ == "__main__":
    main()
