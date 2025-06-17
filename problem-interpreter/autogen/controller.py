from autogen import UserProxyAgent, GroupChat, GroupChatManager
from agents_config import objective_agent, classifier_agent, target_eval_agent
import json

config_list = [
    {
        "model": "llama3:latest",
        "api_type": "ollama",
        "stream": True,
        "client_host": "http://localhost:11434",
    }
]
llm_config = {
    "config_list": config_list
    }

def custom_select_speaker(self, last_speaker, groupchat):
    """Custom speaker selection based on function call."""
    last_message = groupchat.groupchat.messages[-1] if groupchat.groupchat.messages else {}
    content = last_message.get("content", "")
    print(f"[Speaker Selection] Last message: {content}")  # Debug
    try:
        content_json = json.loads(content)
        if isinstance(content_json, dict) and "function" in content_json:
            func_name = content_json["function"]
            print(f"[Speaker Selection] Detected function: {func_name}")  # Debug
            # Map function to agent
            if func_name == "generate_objective":
                print(f"[Speaker Selection] Choosing ObjectiveGenerator")
                return objective_agent
            elif func_name == "classify_problem_type":
                print(f"[Speaker Selection] Choosing ProblemClassifier")
                return classifier_agent
            elif func_name == "get_target_variable_and_metrics":
                print(f"[Speaker Selection] Choosing TargetEvaluator")
                return target_eval_agent
    except json.JSONDecodeError:
        print("[Speaker Selection] No valid JSON function call detected")  # Debug
    # Default to next agent
    agents = groupchat.groupchat.agents
    current_idx = agents.index(last_speaker) if last_speaker in agents else -1
    next_speaker = agents[(current_idx + 1) % len(agents)]
    print(f"[Speaker Selection] Defaulting to next speaker: {next_speaker.name}")  # Debug
    return next_speaker

def run_autogen_pipeline(user_prompt, dataset_metadata):
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
    )

    groupchat = GroupChat(
        agents=[user_proxy, objective_agent, classifier_agent, target_eval_agent],
        messages=[],
        max_round=3,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    # Override select_speaker method
    manager.select_speaker = custom_select_speaker.__get__(manager, GroupChatManager)

    # Sequential function calls
    kickoff_msgs = [
        {
            "function": "generate_objective",
            "arguments": {"user_prompt": user_prompt}
        },
        {
            "function": "classify_problem_type",
            "arguments": {"user_prompt": user_prompt}
        },
        {
            "function": "get_target_variable_and_metrics",
            "arguments": {"user_prompt": user_prompt, "metadata": dataset_metadata}
        }
    ]

    results = []
    for msg in kickoff_msgs:
        print(f"[Pipeline] Initiating chat with message: {msg}")  # Debug
        user_proxy.initiate_chat(
            manager,
            message=json.dumps(msg),
            clear_history=(msg == kickoff_msgs[0])  # Clear history only for first call
        )
        # Extract messages for this round
        all_messages = []
        for agent, msg_list in manager.chat_messages.items():
            all_messages.extend(msg_list)
        # Get assistant responses
        assistant_responses = [m["content"] for m in all_messages if m["role"] == "assistant" and m.get("content")]
        if assistant_responses:
            results.append(assistant_responses[-1])
        else:
            results.append(f"[Pipeline] No assistant response for {msg['function']}")  # Debug fallback

    return results

if __name__ == "__main__":
    user_prompt = input("Enter your problem description: ")
    dataset_metadata = input("Enter dataset metadata (e.g., column names, types): ")

    results = run_autogen_pipeline(user_prompt, dataset_metadata)

    print("\n--- Autogen Pipeline Results ---")
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}: {result}")