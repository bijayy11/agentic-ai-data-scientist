from autogen import UserProxyAgent
from autogen.agents import GroupChat, GroupChatManager
from autogen.agents_config import objective_agent, classifier_agent, target_eval_agent

def run_autogen_pipeline(user_prompt, dataset_metadata):
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10
    )

    groupchat = GroupChat(
        agents=[user_proxy, objective_agent, classifier_agent, target_eval_agent],
        messages=[],
        max_round=5
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=objective_agent.llm_config)

    kickoff_msg = f"""User Prompt: {user_prompt}\nDataset Metadata: {dataset_metadata}\nYour goal is to:\n1. Generate the modeling objective.\n2. Classify the problem type.\n3. Identify the target variable and evaluation metrics."""

    user_proxy.initiate_chat(manager, message=kickoff_msg)
