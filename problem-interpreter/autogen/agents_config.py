import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_llm_provider import CustomLlamaAgent
# from autogen import AssistantAgent
from agents.config import get_local_llama3
from agents.objective_generator import generate_objective
from agents.problem_classifier import classify_problem_type
from agents.target_variable_evaluator import get_target_variable_and_metrics
llama3_raw = get_local_llama3()


objective_agent = CustomLlamaAgent(
    name="ObjectiveGenerator",
    model_fn=llama3_raw,
    system_message="You generate concise modeling objectives from user prompts.",
    function_map={"generate_objective": generate_objective}
)

classifier_agent = CustomLlamaAgent(
    name="ProblemClassifier",
    model_fn=llama3_raw,
    system_message="You classify the problem into Binary Classification, Regression, etc.",
    function_map={"classify_problem_type": classify_problem_type}
)

target_eval_agent = CustomLlamaAgent(
    name="TargetEvaluator",
    model_fn=llama3_raw,
    system_message="You identify the target variable and evaluation metrics.",
    function_map={"get_target_variable_and_metrics": get_target_variable_and_metrics}
)