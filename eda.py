import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('loan_data.csv')  # Change the filename as necessary

# Exploring target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='loan_status', data=data, palette='Set2')
plt.title('Distribution of Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

# Check for missing/null values
missing_values = data.isnull().sum()
print("Missing values in each column: \n", missing_values[missing_values > 0])

# Check data types and basic info
print(data.info())

# Identify numerical and categorical features
numerical_features = data.select_dtypes(include=['int', 'float']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
print("Numerical Features: ", numerical_features)
print("Categorical Features: ", categorical_features)

# Checking feature correlations
plt.figure(figsize=(10, 6))
correlation_matrix = data[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize feature-target relationships for numerical features
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y=feature, data=data, palette='Set2')
    plt.title(f'{feature} vs Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel(feature)
    plt.show()

# Visualize feature-target relationships for categorical features
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='loan_status', data=data, palette='Set2')
    plt.title(f'Distribution of {feature} by Loan Status')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(title='Loan Status')
    plt.show()

# Print insights helpful for model selection
class_distribution = data['loan_status'].value_counts(normalize=True)
print("Class Distribution: \n", class_distribution)

# Check for categorical feature levels
for feature in categorical_features:
    print(f"{feature} unique values: {data[feature].unique()}") 

# Print summary statistics for numerical features
print(data[numerical_features].describe()) 

# Check for potential skewness in numerical features
skewness = data[numerical_features].skew()
print("Skewness of numerical features: \n", skewness) 

# Check for outliers in numerical features using IQR
for feature in numerical_features:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))]
    print(f'Number of outliers in {feature}: {outliers.shape[0]}') 

plt.show()