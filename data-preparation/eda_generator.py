# from autogen import AssistantAgent,config_list_from_dotenv
# import subprocess
# import re

# config_list = config_list_from_dotenv(
#     dotenv_file_path=".env",
#     model_api_key_map={"gpt-4o-mini": "OPENAI_API_KEY"}
# )

# for cfg in config_list:
#     cfg["model"] = "gpt-4o-mini"

# llm_config = {
#     "config_list": config_list
# }

# EDA_SYSTEM_MESSAGE = """You are a skilled data scientist. You are given:
# 1. A machine learning objective (e.g. "classify risky and non-risky loan users"),
# 2. The type of ML problem (e.g. classification, regression, clustering),
# 3. The target variable and appropriate evaluation metrics,
# 4. Dataset metadata (data types, column names, value ranges),
# 5. The head of the dataset (first 5 rows as a preview).

# Your task is to generate a detailed Python EDA script (in a single code block) using libraries such as pandas, matplotlib, seaborn, and optionally plotly or sweetviz. The EDA should:

# - Explore target distribution.
# - Check for missing/null values.
# - Check feature correlations and feature-target relationships.
# - Use visualizations appropriate to data type and ML problem.
# - Print insights helpful for model selection, feature engineering, and hyperparameter tuning (e.g. class imbalance, skew, etc.).
# - Automatically detect data types (numerical/categorical) from metadata.
# - Avoid model training — focus only on EDA and interpretation.
# IMPORTANT:
# Return ONLY the Python code in a single code block, and nothing else — no explanation, no markdown headers, no text before or after the code.
# The code should be ready to run in a .py file after pasted in a file.
# """

# eda_generator_agent = AssistantAgent(
#     name="EDAGenerator",
#     llm_config=llm_config,
#     system_message=EDA_SYSTEM_MESSAGE,
# )

# input_json = {
#     "objective": "Classify risky and non-risky loan users",
#     "ml_problem": "classification",
#     "target_variable": "loan_status",
#     "evaluation_metrics": ["ROC AUC", "Precision", "Recall"]
# }

# dataset_metadata = {
#     "columns": {
#         "age": "int",
#         "income": "float",
#         "loan_amount": "float",
#         "credit_score": "float",
#         "employment_status": "str",
#         "previous_defaults": "int",
#         "marital_status": "str",
#         "education_level": "str",
#         "loan_status": "str"
#     }
# }

# dataset_head = [
#     {
#         "age": 35,
#         "income": 45000.0,
#         "loan_amount": 15000.0,
#         "credit_score": 670.0,
#         "employment_status": "Employed",
#         "previous_defaults": 0,
#         "marital_status": "Single",
#         "education_level": "Bachelors",
#         "loan_status": "Non-Risky"
#     },
#     {
#         "age": 42,
#         "income": 52000.0,
#         "loan_amount": 20000.0,
#         "credit_score": 620.0,
#         "employment_status": "Self-employed",
#         "previous_defaults": 1,
#         "marital_status": "Married",
#         "education_level": "Masters",
#         "loan_status": "Risky"
#     }
# ]

# # Combine all input as a message
# user_input = f"""
# Objective: {input_json['objective']}
# Problem Type: {input_json['ml_problem']}
# Target Variable: {input_json['target_variable']}
# Evaluation Metrics: {', '.join(input_json['evaluation_metrics'])}

# Dataset Metadata:
# {dataset_metadata}

# Dataset Head:
# {dataset_head}
# """

# # Run the agent
# messages = [{"role": "user", "content": user_input}]

# # Run agent
# reply = eda_generator_agent.generate_reply(messages)
# print(reply)
# if match:
#     code_only = match.group(1)

#     # Write the code to eda.py
#     with open("eda.py", "w") as f:
#         f.write(code_only)

#     print("EDA script saved to eda.py")
# else:
#     print("No valid Python code block found in the response.")

# try:
#     print("Running eda.py...\n")
#     subprocess.run(["python3", "eda.py"], check=True)
# except subprocess.CalledProcessError as e:
#     print("Error occurred while running eda.py:", e)



from autogen import AssistantAgent, config_list_from_dotenv
import subprocess
import re

# Load config
config_list = config_list_from_dotenv(
    dotenv_file_path=".env",
    model_api_key_map={"gpt-4o-mini": "OPENAI_API_KEY"}
)

for cfg in config_list:
    cfg["model"] = "gpt-4o-mini"

llm_config = {"config_list": config_list}

# Define EDA system message
EDA_SYSTEM_MESSAGE = """You are a skilled data scientist. You are given:
1. A machine learning objective (e.g. "classify risky and non-risky loan users"),
2. The type of ML problem (e.g. classification, regression, clustering),
3. The target variable and appropriate evaluation metrics,
4. Dataset metadata (data types, column names, value ranges),
5. The head of the dataset (first 5 rows as a preview).

Your task is to generate a detailed Python EDA script for multiple possible EDAs(in a single code block) using libraries such as pandas, matplotlib, seaborn, and optionally plotly or sweetviz. The EDA should:

- Explore target distribution.
- Check for missing/null values.
- Check feature correlations and feature-target relationships.
- Use visualizations appropriate to data type and ML problem.
- Print insights helpful for model selection, feature engineering, and hyperparameter tuning (e.g. class imbalance, skew, etc.).
- Automatically detect data types (numerical/categorical) from metadata.
- Avoid model training — focus only on EDA and interpretation.

IMPORTANT:
Return ONLY the Python code in a single code block, and nothing else — no explanation, no markdown headers, no text before or after the code.
The code should be ready to run in a .py file after pasted in a file.
"""

# Create agent
eda_generator_agent = AssistantAgent(
    name="EDAGenerator",
    llm_config=llm_config,
    system_message=EDA_SYSTEM_MESSAGE,
)

# Input JSON
input_json = {
    "objective": "Classify risky and non-risky loan users",
    "ml_problem": "classification",
    "target_variable": "loan_status",
    "evaluation_metrics": ["ROC AUC", "Precision", "Recall"]
}

dataset_metadata = {
    "columns": {
        "age": "int",
        "income": "float",
        "loan_amount": "float",
        "credit_score": "float",
        "employment_status": "str",
        "previous_defaults": "int",
        "marital_status": "str",
        "education_level": "str",
        "loan_status": "str"
    }
}

dataset_head = [
    {
        "age": 35,
        "income": 45000.0,
        "loan_amount": 15000.0,
        "credit_score": 670.0,
        "employment_status": "Employed",
        "previous_defaults": 0,
        "marital_status": "Single",
        "education_level": "Bachelors",
        "loan_status": "Non-Risky"
    },
    {
        "age": 42,
        "income": 52000.0,
        "loan_amount": 20000.0,
        "credit_score": 620.0,
        "employment_status": "Self-employed",
        "previous_defaults": 1,
        "marital_status": "Married",
        "education_level": "Masters",
        "loan_status": "Risky"
    }
]

# Compose agent input
user_input = f"""
Objective: {input_json['objective']}
Problem Type: {input_json['ml_problem']}
Target Variable: {input_json['target_variable']}
Evaluation Metrics: {', '.join(input_json['evaluation_metrics'])}

Dataset Metadata:
{dataset_metadata}

Dataset Head:
{dataset_head}
"""

# Run the agent
messages = [{"role": "user", "content": user_input}]
reply = eda_generator_agent.generate_reply(messages)

# Extract Python code from response
match = re.search(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
if match:
    code_only = match.group(1)
    print("✅ Python code block extracted from response.")
elif "import pandas" in reply or "df =" in reply:
    code_only = reply
    print("⚠️ No code block found, using raw reply as Python script.")
else:
    print("❌ No usable Python code found in reply.")
    exit(1)

# Write eda.py
with open("eda.py", "w") as f:
    f.write(code_only)
print("✅ eda.py written successfully.")

# Generate loan_data.csv to match the sample
import pandas as pd
sample_df = pd.DataFrame(dataset_head)
sample_df.to_csv("loan_data.csv", index=False)
print("📄 loan_data.csv created.")

# Extract requirements
import_lines = [
    line.strip() for line in code_only.splitlines()
    if line.strip().startswith("import") or line.strip().startswith("from")
]
# Extract imported libraries and clean them
libs = set()
for line in import_lines:
    line = line.replace("import ", ",").replace("from ", ",")
    tokens = [t.strip().split()[0].split('.')[0] for t in line.split(',') if t]
    libs.update(tokens)


standard_libs = {"os", "sys", "re", "math", "itertools", "datetime", "collections"}
third_party_libs = sorted(lib for lib in libs if lib not in standard_libs)

with open("requirements.txt", "w") as f:
    f.write("\n".join(third_party_libs))
print("✅ requirements.txt created:", third_party_libs)

# Install and run
try:
    print("\n📦 Installing dependencies...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

    print("\n🚀 Running eda.py...\n")
    subprocess.run(["python3", "eda.py"], check=True)

except subprocess.CalledProcessError as e:
    print("❌ Error occurred:", e)