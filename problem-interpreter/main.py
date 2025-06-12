from agents.objective_generator import generate_objective
from agents.problem_classifier import classify_problem_type
from agents.target_variable_evaluator import get_target_variable_and_metrics
from autogen.controller import run_autogen_pipeline

if __name__ == "__main__":
    prompt = "Predict customer churn based on customer activity logs"
    metadata = {"columns": ["user_id", "last_login", "activity_score", "churn"]}

    print("\n✨ Step 1: Objective")
    objective = generate_objective(prompt)
    print("Objective:", objective)

    print("\n📈 Step 2: Problem Type")
    problem_type = classify_problem_type(prompt)
    print("Problem Type:", problem_type)

    print("\n🔢 Step 3: Target Variable + Evaluation")
    target, metrics = get_target_variable_and_metrics(prompt, metadata)
    print("Target Variable:", target)
    print("Evaluation Metrics:", metrics)

    print("\n🚀 Running AutoGen Controller\n")
    run_autogen_pipeline(prompt, metadata)