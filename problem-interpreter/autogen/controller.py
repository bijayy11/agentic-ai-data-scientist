from agents_config import objective_agent, classifier_agent, target_eval_agent
import json

def user_input(prompt):
    return input(prompt)

def chat_with_agent(agent, function_name, arguments, history=None):
    message = {
        "role": "user",
        "content": json.dumps({
            "function": function_name,
            "arguments": arguments
        })
    }
    messages = (history or []) + [message]
    return agent._generate_response(messages)

def main():
    # Step 1: Get user problem prompt
    user_problem = user_input("Enter your problem description: ")

    history = []

    # Step 2: Objective Generator
    obj_response = chat_with_agent(objective_agent, "generate_objective", {
        "user_prompt": user_problem
    }, history)
    print("\n[Objective Generator Output]:", obj_response)
    history.append({"role": "assistant", "name": objective_agent.name, "content": obj_response})

    # Step 3: Problem Classifier
    cls_response = chat_with_agent(classifier_agent, "classify_problem_type", {
        "objective_response": obj_response
    }, history)
    print("\n[Problem Classifier Output]:", cls_response)
    history.append({"role": "assistant", "name": classifier_agent.name, "content": cls_response})

    # Step 4: Target Evaluator (asks user for metadata)
    metadata = user_input("Please provide metadata (e.g., dataset columns): ")
    tgt_response = chat_with_agent(target_eval_agent, "get_target_variable_and_metrics", {
        "objective_response": obj_response,
        "classifier_response": cls_response,
        "metadata": metadata
    }, history)
    print("\n[Target Evaluator Output]:", tgt_response)
    history.append({"role": "assistant", "name": target_eval_agent.name, "content": tgt_response})

    # Step 5: Refinement Loop
    for round_num in range(2):
        print(f"\n--- Refinement Round {round_num + 1} ---")

        obj_response = chat_with_agent(objective_agent, "generate_objective", {
            "user_prompt": user_problem,
            "feedback": json.dumps({
            "classifier_response": cls_response,
            "target_eval_response": tgt_response
            })

        }, history)
        print("[Refined Objective]:", obj_response)
        history.append({"role": "assistant", "name": objective_agent.name, "content": obj_response})

        cls_response = chat_with_agent(classifier_agent, "classify_problem_type", {
            "objective_response": obj_response
        }, history)
        print("[Refined Classification]:", cls_response)
        history.append({"role": "assistant", "name": classifier_agent.name, "content": cls_response})

        tgt_response = chat_with_agent(target_eval_agent, "get_target_variable_and_metrics", {
            "objective_response": obj_response,
            "classifier_response": cls_response,
            "metadata": metadata
        }, history)
        print("[Refined Target Eval]:", tgt_response)
        history.append({"role": "assistant", "name": target_eval_agent.name, "content": tgt_response})

    # Final JSON output
    final_output = {
        "objective_agent": json.loads(json.dumps(obj_response)),
        "classifier_agent": json.loads(json.dumps(cls_response)),
        "target_eval_agent": json.loads(json.dumps(tgt_response)),
    }
    return final_output




if __name__ == "__main__":
    result = main()
    print("\n=== Final JSON Output ===")
    print(json.dumps(result, indent=2))