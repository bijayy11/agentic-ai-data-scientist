from autogen import AssistantAgent
from models.local_llama3 import get_local_llama3

llm = get_local_llama3()

objective_agent = AssistantAgent(
    name="ObjectiveGenerator",
    llm_config={"llm": llm},
    system_message="You generate concise modeling objectives from user prompts."
)

classifier_agent = AssistantAgent(
    name="ProblemClassifier",
    llm_config={"llm": llm},
    system_message="You classify the problem into Binary Classification, Regression, etc."
)

target_eval_agent = AssistantAgent(
    name="TargetEvaluator",
    llm_config={"llm": llm},
    system_message="You identify the target variable and evaluation metrics."
)
