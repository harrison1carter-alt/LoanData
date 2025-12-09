import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
import pandas as pd

print("Please select your CSV file to upload:")
uploaded = files.upload()

for fn in uploaded.keys():
  print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
  # To load it into a pandas DataFrame, you can do:
  # df = pd.read_csv(io.BytesIO(uploaded[fn]))
  # For now, let's assume the uploaded file is the one you need to replace your current 'df'
  # You would then load it as a new DataFrame or assign it to your existing 'df'
  # For demonstration, I'll load it into a new df_uploaded variable
  import io
  df_uploaded = pd.read_csv(io.BytesIO(uploaded[fn]))
  print(f"Successfully loaded '{fn}' into df_uploaded.")
  display(df_uploaded.head())

# If you want to use this as your main df, you'd then do:
# df = df_uploaded.copy()
# print("Updated the main DataFrame 'df' with the uploaded data.")

# Assign the uploaded DataFrame to the main DataFrame 'df'
df = df_uploaded.copy()
print("Main DataFrame 'df' updated with the uploaded data.")
display(df.head())

# Check first 5 rows of dataframe
display(df.head())

# Drop columns that have no variation or are unique

# Identify columns with no variation (all values are the same)
constant_columns = [col for col in df.columns if df[col].nunique() == 1]

# Identify columns with unique values (all values are distinct)
unique_columns = [col for col in df.columns if df[col].nunique() == len(df)]

# Combine the lists of columns to drop
columns_to_drop = list(set(constant_columns + unique_columns))

# Drop the identified columns
df = df.drop(columns=columns_to_drop)

print(f"Dropped constant columns: {constant_columns}")
print(f"Dropped unique columns: {unique_columns}")
print(f"New DataFrame shape: {df.shape}")

# Display the first few rows of the updated DataFrame
display(df.head())

# Get a concise summary of the DataFrame, including data types and non-null values
print("\n--- DataFrame Info ---")
df.info()

# Get descriptive statistics for numerical columns
print("\n--- Descriptive Statistics for Numerical Columns ---")
display(df.describe())

# Get descriptive statistics for categorical columns (object type)
print("\n--- Descriptive Statistics for Categorical Columns ---")
display(df.describe(include='object'))

# Check unique values and their counts for categorical columns for a deeper understanding
print("\n--- Unique Values and Counts for Categorical Columns ---")
for col in df.select_dtypes(include='object').columns:
    print(f"\nColumn '{col}':")
    print(df[col].value_counts())
    print(f"Number of unique values: {df[col].nunique()}")

# Inspect missing values
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})

missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

print("--- Missing Values in DataFrame ---")
display(missing_df)

# Apply imputation as instructed in the instructions pdf (create copies so original rows can be restored if needed)

# Create a copy of the DataFrame
df_imputed = df.copy()

# Impute FICO_score and Monthly_Gross_Income with their medians
df_imputed['FICO_score'].fillna(df_imputed['FICO_score'].median(), inplace=True)
df_imputed['Monthly_Gross_Income'].fillna(df_imputed['Monthly_Gross_Income'].median(), inplace=True)

# Impute Employment_Sector with its mode
df_imputed['Employment_Sector'].fillna(df_imputed['Employment_Sector'].mode()[0], inplace=True)

# Verify that missing values have been handled
print("--- Missing Values After Imputation ---")
display(df_imputed.isnull().sum())

df = df_imputed.copy() # Update df with the imputed data for subsequent steps

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

numeric_cols = ['Granted_Loan_Amount', 'Requested_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment']

# Boxplots to highlight outliers for numerical columns
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col} Before Outlier Removal')
    plt.ylabel(col)
    plt.show()

# Remove Outliers using Z-score

from scipy import stats
import numpy as np

# Compute Z-scores for only numeric columns
z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))

# Choose threshold
threshold = 3  # common choice: 3 standard deviations

# Identify rows to keep (all Z-scores <= threshold)
rows_to_keep = (z_scores < threshold).all(axis=1)

# Track counts before/after
before_count = df.shape[0]
df_clean = df[rows_to_keep].copy()
after_count = df_clean.shape[0]

print(f"Outlier removal complete:")
print(f"Rows before: {before_count}")
print(f"Rows after:  {after_count}")
print(f"Rows removed: {before_count - after_count}")

# BoxPlots after outlier removal

for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df_clean[col])
    plt.title(f'Box Plot of {col} After Outlier Removal')
    plt.ylabel(col)
    plt.show()

# Correlation between numerical features
numeric_cols = ['FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment', 'Granted_Loan_Amount','Requested_Loan_Amount', 'Approved']

# Plot correlation matrix, identify highly correlated pairs automatically (|corr|>0.95, excluding self-correlation)
corr = df_clean[numeric_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Numeric Correlation Heatmap (with strongly correlated column)")
plt.show()

# Correlation between Numeric and Categorical Variables (Correlation Ratio / η²)
from scipy import stats

def correlation_ratio(categories, values):
    categories = np.array(categories)
    values = np.array(values)
    cat_levels = np.unique(categories)
    overall_mean = np.mean(values)
    numerator = sum(len(values[categories == cat]) *
                    (np.mean(values[categories == cat]) - overall_mean) ** 2
                    for cat in cat_levels)
    denominator = sum((values - overall_mean) ** 2)
    return np.sqrt(numerator / denominator) if denominator != 0 else 0

# test all numeric–categorical pairs
num_cat_results = []


# Identify categorical and numerical columns
categorical_cols_clean = df_clean.select_dtypes(include='object').columns.tolist()
numeric_cols_clean = df_clean.select_dtypes(include=np.number).columns.tolist()


numeric_cols_for_eta = [col for col in numeric_cols_clean if col not in ['bounty', 'Approved', ]]
categorical_cols_for_eta = [col for col in categorical_cols_clean]


for num_col in numeric_cols_for_eta:
    for cat_col in categorical_cols_for_eta:
        eta = correlation_ratio(df_clean[cat_col], df_clean[num_col])
        num_cat_results.append((num_col, cat_col, eta))

num_cat_results = sorted(num_cat_results, key=lambda x: x[2], reverse=True)
print("Top 10 Numeric-Categorical Correlation Ratios (η²):")
for num_col, cat_col, eta in num_cat_results[:10]:
    print(f"{num_col} - {cat_col}: η² = {eta**2:.3f} (η = {eta:.3f})")


# Reorganize the correlation ratio results into a pivot table/matrix for heatmap
eta_matrix = pd.DataFrame(num_cat_results, columns=['Numerical_Feature', 'Categorical_Feature', 'Eta'])
eta_pivot = eta_matrix.pivot(index='Numerical_Feature', columns='Categorical_Feature', values='Eta')

# Plot the heatmap of correlation ratios (Eta)
plt.figure(figsize=(12, 8))
sns.heatmap(eta_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Heatmap of Correlation Ratios (η) between Numeric and Categorical Features")
plt.xlabel("Categorical Features")
plt.ylabel("Numerical Features")
plt.tight_layout()
plt.show()

# Plot 'Approved' to visualise the count and balance
plt.figure(figsize=(8, 6))
sns.countplot(x='Approved', data=df_clean, palette='viridis')
plt.title('Distribution of Loan Approval Status')
plt.xlabel('Approved (0 = Denied, 1 = Approved)')
plt.ylabel('Number of Applications')
plt.xticks([0, 1], ['Denied', 'Approved'])
plt.show()

# Display the count and percentage for balance assessment
approval_counts = df_clean['Approved'].value_counts()
approval_percentages = df_clean['Approved'].value_counts(normalize=True) * 100

print("\n--- Loan Approval Status Counts ---")
display(approval_counts)
print("\n--- Loan Approval Status Percentages ---")
display(approval_percentages)

# Visualise numerical variables against target variable

numerical_cols = ['Granted_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment']

for col in numerical_cols:
    plt.figure(figsize=(12,5))
    sns.histplot(
        data=df,
        x=col,
        hue='Approved',
        kde=True,
        stat='density',
        common_norm=False,
        palette='coolwarm',
        alpha=0.6
    )
    plt.title(f'Distribution of {col} by Approval Status')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    sns.boxplot(
        data=df,
        x='Approved',
        y=col,
        hue='Approved',
        palette='coolwarm',
        legend=False
    )
    plt.title(f'Box Plot of {col} by Approval Status')
    plt.xlabel('Approved (0 = Denied, 1 = Approved)')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Visualise categorical variables against target variable

categorical_cols = [
    'Reason', 'Fico_Score_group', 'Employment_Status',
    'Employment_Sector', 'Lender', 'Ever_Bankrupt_or_Foreclose'
]

for col in categorical_cols:
    approval_rates = df.groupby(col)['Approved'].mean().sort_values(ascending=False) * 100

    plt.figure(figsize=(10,5))
    sns.barplot(x=approval_rates.index, y=approval_rates.values, color='skyblue')
    plt.title(f'Approval Rate by {col}')
    plt.xlabel(col)
    plt.ylabel('Approval Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print(f"\nApproval Rate by {col}:\n{approval_rates.round(2)}")

# Cross-feature plot: FICO_score vs. Approved, separated by Lender
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_clean, x='Lender', y='FICO_score', hue='Approved', palette=['red', 'green'])
plt.title('FICO Score Distribution by Lender and Approval Status')
plt.xlabel('Lender')
plt.ylabel('FICO Score')
plt.legend(title='Approved', labels=['Denied', 'Approved'])
plt.tight_layout()
plt.show()

# Another cross-feature plot: Monthly_Gross_Income vs. Approved, separated by Lender
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_clean, x='Lender', y='Monthly_Gross_Income', hue='Approved', palette=['red', 'green'])
plt.title('Monthly Gross Income Distribution by Lender and Approval Status')
plt.xlabel('Lender')
plt.ylabel('Monthly Gross Income')
plt.legend(title='Approved', labels=['Denied', 'Approved'])
plt.tight_layout()
plt.show()

# Split dataset into train and test
from sklearn.model_selection import train_test_split
import pandas as pd

# Rename your df to df_model.
df_model = df_clean.copy()

# Define target variable
y = df_model['Approved']

# Drop target variable and corelated variables from features
X = df_model.drop(columns=['Approved', 'bounty'])

# Identify categorical features for encoding
categorical_features = X.select_dtypes(include='object').columns

# Encode categorical features using one-hot encoding
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Set RANDOM_STATE = 42 for reproducibility.
random_state = 42

# Add your code here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

from sklearn.linear_model import LogisticRegression

# Train logistic regression model
logistic_model = LogisticRegression(random_state=random_state, solver='liblinear', max_iter=1000)
logistic_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predicted probabilities
y_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]

# Set your custom threshold
threshold = 0.3

# Apply threshold to get new predicted classes
y_pred_logistic_thresh = (y_proba_logistic >= threshold).astype(int)

# Calculate metrics
accuracy_logistic = accuracy_score(y_test, y_pred_logistic_thresh)
precision_logistic = precision_score(y_test, y_pred_logistic_thresh)
recall_logistic = recall_score(y_test, y_pred_logistic_thresh)
f1_logistic = f1_score(y_test, y_pred_logistic_thresh)
roc_auc_logistic = roc_auc_score(y_test, y_proba_logistic)   # AUC uses probabilities, not thresholded predictions

print(f"--- Logistic Regression Model Evaluation (Threshold = {threshold}) ---")
print(f"Accuracy:  {accuracy_logistic:.4f}")
print(f"Precision: {precision_logistic:.4f}")
print(f"Recall:    {recall_logistic:.4f}")
print(f"F1-Score:  {f1_logistic:.4f}")
print(f"ROC-AUC:   {roc_auc_logistic:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_logistic_thresh))

# Plot Confusion Matrix
cm_logistic = confusion_matrix(y_test, y_pred_logistic_thresh)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_logistic, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold = {threshold})')
plt.show()

from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=random_state)
decision_tree_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred_tree = decision_tree_model.predict(X_test)
y_proba_tree = decision_tree_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)
roc_auc_tree = roc_auc_score(y_test, y_proba_tree)

print("--- Decision Tree Model Evaluation ---")
print(f"Accuracy: {accuracy_tree:.4f}")
print(f"Precision: {precision_tree:.4f}")
print(f"Recall: {recall_tree:.4f}")
print(f"F1-Score: {f1_tree:.4f}")
print(f"ROC-AUC: {roc_auc_tree:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_tree))

# Plot Confusion Matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Denied', 'Approved'], yticklabels=['Denied', 'Approved'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Decision Tree')
plt.show()

# Variable importance for Decision Tree
feature_importances = decision_tree_model.feature_importances_
feature_names = X_train.columns

# Create a pandas Series for better visualization
importance_df = pd.Series(feature_importances, index=feature_names)

# Sort the features by importance in descending order
importance_df = importance_df.sort_values(ascending=False)

# Print top features
print("--- Decision Tree Feature Importances ---")
display(importance_df.head(10)) # Display top 10 features

model = decision_tree_model

# Save your model as .pkl file for streamlit app development
import pickle

filename = 'my_model.pkl'  # Choose a path and descriptive filename with .pkl extension

# Open the file in binary write mode ('wb')
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved successfully to {filename}")

