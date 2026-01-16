import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use("seaborn")

df = pd.read_csv('https://raw.githubusercontent.com/GUC-DM/W2025/refs/heads/main/data/loan_data.csv')
df.head()

# general info about dataset: columns, non-null counts, and data types
df.info()

# Identify numeric columns and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Correlation matrix for numeric columns
corr = df.corr(numeric_only=True)

df.drop_duplicates(inplace=True) # Remove duplicate rows

df.columns = df.columns.str.strip().str.replace(" ", "_") # Clean column names

# Define a function to convert numeric strings to numeric values
def clean_numeric(x):
    if isinstance(x, str):  # Check if the value is a string
        x = x.replace("$", "").replace(",", "").replace("%", "").strip()  # Remove currency symbols, commas, percent signs, and extra spaces
    return pd.to_numeric(x, errors="coerce")  # Convert to numeric, set invalid values to NaN

# List of object columns that should be numeric
numeric_string_cols = ['AnnualIncome', 'LoanAmount', 'LoanDuration', 'MonthlyLoanPayment', 'MonthlyIncome', 'CreditScore',
                       'NumberOfDependents','JobTenure','BankruptcyHistory','PreviousLoanDefaults']

# Apply the numeric conversion function to the listed columns if they exist in the dataset
for col in numeric_string_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# Separate numeric and categorical columns for further processing
num_cols = df.select_dtypes(include=['int64', 'float64']).columns  # Numeric columns
cat_cols = df.select_dtypes(include=['object']).columns            # Categorical columns

# Impute missing numeric values using the median of each column
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Impute missing categorical values with the most frequent value (mode) and clean the text
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])  # Fill missing values with mode
    df[col] = df[col].astype(str).str.strip().str.title()  # Remove extra spaces and capitalize words

# Automatically encode binary categorical columns (Yes/No, True/False, Y/N)
binary_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "y": 1, "n": 0}

for col in cat_cols:
    col_lower = df[col].astype(str).str.lower().str.strip()  # Convert text to lowercase and remove spaces
    df[col] = col_lower.map(binary_map).fillna(df[col])  # Map known binary values to 0/1, leave other values unchanged

# If MonthlyIncome column is missing but AnnualIncome exists, calculate MonthlyIncome
if "MonthlyIncome" not in df.columns and "AnnualIncome" in df.columns:
    df["MonthlyIncome"] = df["AnnualIncome"] / 12  # Divide annual income by 12 to get monthly income

# Ensure key numeric columns are of numeric type for calculations
for col in ["MonthlyIncome", "MonthlyLoanPayment"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, invalid values become NaN

df.head() # Show the rows of the cleaned dataset

sns.set_style("whitegrid") # Set the background style of the plot

df['LoanApproved_num'] = df['LoanApproved'].map({'Yes':1, 'No':0}) # Encode LoanApproved to numerical values

approval_rates = df.groupby('EducationLevel')['LoanApproved'].mean().sort_values(ascending=False) # Group by EducationLevel and calculate the mean approval rate

# compute the average of the numeric LoanApproved values for each education level
# sort the approval rates in descending order
plt.figure(figsize=(10, 6)) # Plot the bar chart

# X-axis: Education levels, Y-axis: Corresponding average approval rates, hue uses education level, Color palette used for bars, Hide the legend
sns.barplot(x=approval_rates.index, y=approval_rates.values, hue=approval_rates.index, palette='magma', legend=False)

plt.title('Average Loan Approval Rate by Education Level') # Set the plot title

plt.xlabel('Education Level')  # Set X-axis label

plt.ylabel('Approval Rate') # Set Y-axis label

plt.xticks(rotation=45) # Rotate X-axis labels for better readability

plt.show() # Show the plot

sns.set_style("whitegrid")  # Set the background style of the plot

approved_df = df[df['LoanApproved'] == 1].copy() # Filter approved applicants

if approved_df.empty:  # If there are no approved applicants in the dataset
    print("No approved applicants in the dataset!")  # Print a warning message
else:
    quartiles = approved_df['AnnualIncome'].quantile([0.25, 0.5, 0.75])  # Calculate 25th, 50th (median), and 75th percentiles
    print("Quartiles of Annual Income among approved applicants:")  # Print heading
    print(quartiles)  # Show the calculated quartiles

    plt.figure(figsize=(8, 6))  # Set the figure size for the plot
    sns.boxplot(y=approved_df['AnnualIncome'], color='purple')  # Create a vertical boxplot of AnnualIncome with purple color
    plt.title('Annual Income Distribution Among Approved Applicants')  # Add a title to the plot
    plt.ylabel('Annual Income $')  # Label the y-axis
    plt.show()  # Display the boxplot

sns.set_style("whitegrid") # Set the background style of plots to a white grid

df['Age'] = pd.to_numeric(df['Age'], errors='coerce') # Convert Age to numeric, invalid values become NaN

df['CreditScore'] = pd.to_numeric(df['CreditScore'], errors='coerce') # Convert CreditScore to numeric, invalid values become NaN

df = df.dropna(subset=['Age', 'CreditScore']) # Remove any rows where Age or CreditScore is missing (NaN)

correlation = df['Age'].corr(df['CreditScore'])  # Calculate the correlation between Age and CreditScore
print("Correlation between Age and CreditScore:", correlation) #Print correlation

# Choose columns for the x-axis and y-axis, set the figure size, make the scatter points pink, make the regression line red
sns.lmplot(x='Age', y='CreditScore', data=df, height=6, aspect=1.5,scatter_kws={'color': 'pink'}, line_kws={'color': 'red'})

plt.title('Relationship Between Age and Credit Score') # Add a title to the plot

plt.xlabel('Age') # Label the x-axis

plt.ylabel('Credit Score') # Label the y-axis

plt.show() # Show the final plot

plt.figure(figsize=(12, 7))  # Create a figure of size 12x7 inches
sns.set_style("whitegrid")    # Set a clean white grid background for the plot

print(df['MonthlyIncome'].skew()) #show skewness

# Plot histogram using KDE which is kernel density estimate
sns.histplot( df['MonthlyIncome'], kde=True, bins=40, color='purple')

plt.title('Histogram of Monthly Income with KDE')  # Add a title to the plot
plt.xlabel('Monthly Income')                        # Label for x-axis
plt.ylabel('Frequency')                             # Label for y-axis

plt.axvline(df['MonthlyIncome'].mean(), color='black', linestyle='--', label='Mean')  # Mean line in black
plt.axvline(df['MonthlyIncome'].median(), color='blue', linestyle='--', label='Median')  # Median line in blue

plt.legend()  # Show the legend showing which line is mean and median
plt.show()    # Show the plot

from sklearn.model_selection import train_test_split  # Import function to split dataset into training and testing sets

df_encoded = df.copy()  # Make a copy of the cleaned and preprocessed dataset to avoid modifying the original

y = df['LoanApproved'] # Extract the target variable LoanApproved for modeling

X = df_encoded.drop('LoanApproved', axis=1)  # Remove the target column from features to create X

print("Missing in X:", X.isnull().sum().sum())  # Total missing values in features
print("Missing in y:", y.isnull().sum())       # Total missing values in target


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Use 20% of data for testing, stratify ensures target proportion
)

print(f"Training set: {X_train.shape[0]} samples")  # Number of samples in the training set
print(f"Testing set: {X_test.shape[0]} samples")    # Number of samples in the testing set
print("Number of features:", X_train.shape[1])      # Number of features used for modeling

from sklearn.tree import DecisionTreeClassifier

categorical_cols = X_train.select_dtypes(include=['object']).columns # Identify categorical columns in the training set

# One-hot encode categorical columns
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True) # Convert categorical features to binary columns
X_test_encoded  = pd.get_dummies(X_test,  columns=categorical_cols, drop_first=True)

X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0) # Match test to train columns


y_notnull = y_train.notnull() # Remove rows where y_train is NaN, True for rows where y_train is not NaN

X_train_encoded_clean = X_train_encoded[y_notnull] # Keep only rows in X_train corresponding to non-missing target
y_train_clean         = y_train[y_notnull] # Keep only non missing target values

# Train decision tree
dt_clf = DecisionTreeClassifier(random_state=42) # Create classifier with a fixed random state for reproducibility
dt_clf.fit(X_train_encoded_clean, y_train_clean) # Train the classifier on cleaned and encoded training data

y_pred = dt_clf.predict(X_test_encoded) # Predict target values for test features

y_test_notnull = y_test.notnull()  #Clean y_test and align X_test
y_test_clean  = y_test[y_test_notnull]  # Keep only the non-missing values in y_test
X_test_clean = X_test_encoded[y_test_notnull]  # Filter features using the same mask
y_pred_clean = dt_clf.predict(X_test_clean) #predictions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay  # Import evaluation metrics
import matplotlib.pyplot as plt  # Import plotting library for visualization
from sklearn.tree import export_graphviz  # Import function to export decision tree to DOT format
import graphviz  # Import Graphviz to visualize decision trees

test_accuracy = accuracy_score(y_test_clean, y_pred_clean)  # Accuracy overall correctness
test_precision = precision_score(y_test_clean, y_pred_clean)  # Precision all predicted positives
test_recall = recall_score(y_test_clean, y_pred_clean)  # Recall correct positive predictions
test_f1 = f1_score(y_test_clean, y_pred_clean)  # F1-Score mean of precision and recall

# Training accuracy
y_train_pred = dt_clf.predict(X_train_encoded)
train_accuracy = accuracy_score(y_train, y_train_pred)

overfit_percent = (train_accuracy - test_accuracy) * 100 # Calculating overfitting percentage which is difference between training and test accuracy


print(f"Accuracy: {accuracy_score(y_test_clean, y_pred_clean):.2f}")  # show accuracy
print(f"Precision: {precision_score(y_test_clean, y_pred_clean):.2f}")  # show precision
print(f"Recall: {recall_score(y_test_clean, y_pred_clean):.2f}")  # show recall
print(f"F1-Score: {f1_score(y_test_clean, y_pred_clean):.2f}")  # show F1-score
print(f"Training Accuracy: {train_accuracy:.2f}")  # show training accuracy
print(f"Test Accuracy: {test_accuracy:.2f}")  # show test accuracy
print(f"Overfitting: {overfit_percent:.2f}%")  # show overfitting percentage

cm = confusion_matrix(y_test_clean, y_pred_clean)  # Create confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Approved', 'Approved'])  # Set labels
disp.plot(cmap='Purples')  # Plot confusion matrix with a purple color
plt.show()  # Show the plot

# Decision Tree Visualization with Graphviz, Export decision tree to DOT format
dot_data = export_graphviz(
    dt_clf,  # Trained decision tree classifier
    out_file=None,  # keep in memory
    feature_names=X_train_encoded.columns,  # Names of features used in the tree
    class_names=['Not Approved', 'Approved'],  # Class labels
    filled=True,  # Fill nodes with colors to indicate class
    rounded=True,  # Rounded node boxes
    special_characters=True  # Allow special characters in labels
)

graph = graphviz.Source(dot_data) # Create graph

graph  # Show the decision tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train a Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_encoded_clean, y_train_clean)

# Predict on the test set
y_pred_rf = rf_clf.predict(X_test_clean)

# Evaluate the Random Forest
accuracy_rf = accuracy_score(y_test_clean, y_pred_rf)
precision_rf = precision_score(y_test_clean, y_pred_rf)
recall_rf = recall_score(y_test_clean, y_pred_rf)
f1_rf = f1_score(y_test_clean, y_pred_rf)

print("Random Forest Performance:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1-Score: {f1_rf:.2f}")

# Compare with Decision Tree performance
print("\nDecision Tree Performance old:")
print(f"Accuracy: {accuracy_score(y_test_clean, y_pred_clean):.2f}")
print(f"Precision: {precision_score(y_test_clean, y_pred_clean):.2f}")
print(f"Recall: {recall_score(y_test_clean, y_pred_clean):.2f}")
print(f"F1-Score: {f1_score(y_test_clean, y_pred_clean):.2f}")

