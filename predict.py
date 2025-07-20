import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import chi2_contingency, pearsonr
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import plotly.express as px
from flask import Flask, render_template, request
import joblib
import plotly.express as px


# Load dataset
df = pd.read_csv("healthcare_dataset.csv")


# DATA CLEANING AND ENCODING
df.drop(columns=['Patient_ID', 'Name', 'Address'], errors='ignore', inplace=True)


# Convert Recovery_Time and Treatment_Duration to numeric (in days)
def convert_duration(duration):
    if pd.isnull(duration):
        return np.nan
    duration = duration.lower().strip()
    if 'day' in duration:
        return int(duration.split()[0])
    elif 'week' in duration:
        return int(duration.split()[0]) * 7
    elif 'month' in duration:
        return int(duration.split()[0]) * 30
    elif 'year' in duration:
        return int(duration.split()[0]) * 365
    return np.nan


# New columns with the converted Time
df['Recovery_Days'] = df['Recovery_Time'].apply(convert_duration)
df['Treatment_Days'] = df['Treatment_Duration'].apply(convert_duration)

# Extract systolic and diastolic from Blood_Pressure
df[['Systolic', 'Diastolic']] = df['Blood_Pressure'].str.split('/', expand=True).astype(float)


# Create Age_Group column
def age_group(age):
    if age < 18:
        return 'Child'
    elif age < 40:
        return 'Young Adult'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'


# New column named 'Age_Group'
df['Age_Group'] = df['Age'].apply(age_group)

# Convert 'FamilyHistory' to binary: 'Yes' -> 1, 'No' -> 0
df['FamilyHistory'] = df['FamilyHistory'].map({'Yes': 1, 'No': 0})

# Create a binary column for hypertension risk
df['Hypertension_Risk'] = np.where((df['Systolic'] >= 130) | (df['Diastolic'] >= 80), 1, 0)

# Drop the original 'Blood_Pressure' column
df.drop('Blood_Pressure', axis=1, inplace=True)

# Convert 'SAT' to numeric, coerce errors to NaN
df['SAT'] = pd.to_numeric(df['SAT'], errors='coerce')

# Fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
df['X-ray_Results'] = df['X-ray_Results'].map({'Normal': 0, 'Abnormal': 1})
df['Allergies'] = df['Allergies'].map({'No': 0, 'Yes': 1})


# Fill missing values with median for numeric columns
df = df.fillna(df.median(numeric_only=True))


# Handling Outliers - Using IQR
def handle_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    median = df[column].median()
    df[column] = np.where((df[column] < lower) | (df[column] > upper), median, df[column])


for col in numeric_cols:
    handle_outliers(col)


# Define scale_cols
scale_cols = ['Age', 'Heart_Rate', 'Temperature', 'SAT', 'Treatment_Days', 'Recovery_Days', 'Systolic', 'Diastolic']

# Standardize the scale_cols
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[scale_cols]), columns=[col + '_scaled' for col in scale_cols])
df = pd.concat([df, df_scaled], axis=1)

def get_aggregated_insights():
    top_diag = df['Diagnosis'].value_counts().head(5).index.tolist()
    common_meds = df['Medication'].value_counts().head(3).index.tolist()
    avg_recovery = round(df['Recovery_Days'].mean(), 1)

    top_satisfaction_row = df.loc[df['SAT'].idxmax()]
    best_doctor = top_satisfaction_row['Doctor_Name']
    best_hospital = top_satisfaction_row['Hospital_Name']
    surgery_counts = df['Surgery_Type'].value_counts()

    longest_recovery = df.groupby('Surgery_Type')['Recovery_Days'].mean().sort_values(ascending=False).head(2).index.tolist()

    return {
        'top_diagnoses': top_diag,
        'common_meds': common_meds,
        'Surgery Types': surgery_counts,
        'avg_recovery': avg_recovery,
        'best_doctor': best_doctor,
        'best_hospital': best_hospital,
        'longest_recovery': longest_recovery
    }


# Set default Seaborn theme and palette
sns.set(style="whitegrid")
sns.set_palette("pastel")

# EDA
# Distribution of Age, Treatment Days, Recovery Days, and SAT
df[['Age', 'Treatment_Days', 'Recovery_Days', 'SAT']].hist(figsize=(10, 6), bins=20)
plt.suptitle("Distributions of Age, Treatment & Recovery Days, and SAT")
plt.tight_layout()
plt.show()

# Heart Rate vs Diagnosis
top10_diags = df['Diagnosis'].value_counts().head(10).index.tolist()
filtered_df = df[df['Diagnosis'].isin(top10_diags)]
plt.figure(figsize=(12, 5))
sns.boxplot(x='Diagnosis', y='Heart_Rate', data=filtered_df)
plt.xticks(rotation=45)
plt.title("Heart Rate across Top 10 Diagnoses")
plt.tight_layout()
plt.show()

# Temperature vs Diagnosis
plt.figure(figsize=(12, 5))
sns.violinplot(x='Diagnosis', y='Temperature', data=filtered_df)
plt.xticks(rotation=45)
plt.title("Temperature Distribution by Top 10 Diagnosis Types")
plt.tight_layout()
plt.show()


# Correlation heatmap
num_cols = ['Age', 'Systolic', 'Diastolic', 'Heart_Rate', 'Temperature', 'Treatment_Days', 'Recovery_Days', 'SAT', 'FamilyHistory']
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm'); plt.title("Correlation Heatmap"); plt.show()

# Scatterplot: Treatment vs Recovery
sns.scatterplot(data=df, x='Treatment_Days', y='Recovery_Days', hue='Age_Group')
plt.title("Treatment Duration vs Recovery Time by Age Group")
plt.xlabel("Treatment Days")
plt.ylabel("Recovery Days")
plt.show()

# Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Gender', palette='Set2')
plt.title("Patient Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Diagnosis Distribution (Top 10)
top_diagnoses = df['Diagnosis'].value_counts().head(10)
print("\nTop 10 Diagnoses:")
print(top_diagnoses)

# Boxplot: SAT Score by Age Group
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='Age_Group', y='SAT', palette='Pastel1')
plt.title("SAT Scores by Age Group")
plt.ylabel("Satisfaction Score")
plt.show()

# Age distribution by Gender
plt.figure(figsize=(10, 6))
sns.violinplot(x='Gender', y='Age', data=df, inner='quartile', palette='pastel')
plt.title('Age Distribution by Gender')
plt.show()
df.head()


# Average Recovery Time by Diagnosis (Top 10)
top_diagnoses_index = df['Diagnosis'].value_counts().head(10).index
avg_recovery = df[df['Diagnosis'].isin(top_diagnoses_index)].groupby('Diagnosis')['Recovery_Days'].mean().sort_values()
plt.figure(figsize=(10,5))
sns.barplot(x=avg_recovery.values, y=avg_recovery.index, palette='Blues_d')
plt.title("Average Recovery Time for Top 10 Diagnoses")
plt.xlabel("Average Recovery Days")
plt.show()

# Interactive bar chart for SAT by Doctor
def SATDoc_chart():
    avg_sat = df.groupby('Doctor_Name')['SAT'].mean().sort_values().reset_index()
    fig = px.bar(avg_sat, x='Doctor_Name', y='SAT', title='Average SAT per Doctor')
    fig.update_layout(xaxis={'categoryorder': 'total ascending'})
    return fig.to_html(full_html=False)


# Interactive bar chart for Recovery Time by Hospital
def RecHos_chart():
    recovery_by_hospital = df.groupby('Hospital_Name')['Recovery_Days'].mean().sort_values().reset_index()
    fig = px.bar(recovery_by_hospital, x='Hospital_Name', y='Recovery_Days',
                 title='Avg Recovery Time by Hospital', color='Recovery_Days')
    fig.update_layout(xaxis={'categoryorder': 'total ascending'})
    return fig.to_html(full_html=False)

# Get top 10 most frequent Diagnosis-Medication pairs
top10 = df.groupby(['Diagnosis', 'Medication']).size().sort_values(ascending=False).head(10).reset_index().drop(columns=0)
avg_recovery = df.groupby(['Diagnosis', 'Medication'])['Recovery_Days'].mean().reset_index()
merged = pd.merge(top10, avg_recovery, on=['Diagnosis', 'Medication'])
pivot = merged.pivot(index='Diagnosis', columns='Medication', values='Recovery_Days')
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap='Blues')
plt.title("Avg Recovery Days for Top Diagnosis-Medication Pairs")
plt.tight_layout()
plt.show()

# 1. Does Family History or Allergies Influence Diagnosis?
top_diagnoses = df['Diagnosis'].value_counts().head(5).index
df_top_diag = df[df['Diagnosis'].isin(top_diagnoses)]

# -- Family History vs Diagnosis
family_crosstab = pd.crosstab(df_top_diag['FamilyHistory'], df_top_diag['Diagnosis'])
print("\nFamily History vs Diagnosis:")
print(family_crosstab)

if not family_crosstab.empty:
    chi2, p_fam, _, _ = chi2_contingency(family_crosstab)
    print(f"Chi-square test (Family History): p = {p_fam:.4f}")
    family_crosstab.plot(kind='bar', stacked=True, figsize=(8,5), colormap='viridis')
    plt.title('Top Diagnoses by Family History')
    plt.xlabel('Family History')
    plt.ylabel('Count')
    plt.show()
else:
    print("No data available for Family History vs Diagnosis analysis.")

# -- Allergies vs Diagnosis
df_allergy_valid = df_top_diag[df_top_diag['Allergies'].notna() & (df_top_diag['Allergies'] != '')]
allergy_crosstab = pd.crosstab(df_allergy_valid['Allergies'], df_allergy_valid['Diagnosis'])
print("\nAllergies vs Diagnosis:")
print(allergy_crosstab)

if not allergy_crosstab.empty:
    chi2, p_allergy, _, _ = chi2_contingency(allergy_crosstab)
    print(f"Chi-square test (Allergies): p = {p_allergy:.4f}")
    allergy_crosstab.plot(kind='bar', stacked=True, figsize=(8,5), colormap='plasma')
    plt.title('Top Diagnoses by Allergy Status')
    plt.xlabel('Allergies')
    plt.ylabel('Count')
    plt.show()
else:
    print("No valid data found for Allergies vs Diagnosis. Skipping analysis.")


# 2. Do Some Surgeries Take Longer to Treat and Recover From?
surgery_avg = df.groupby('Surgery_Type')[['Treatment_Days', 'Recovery_Days']].mean().sort_values('Treatment_Days', ascending=False)
print("\nAverage Treatment and Recovery Days by Surgery Type:")
print(surgery_avg.head())

# Boxplot for Top 5 Surgery Types
top_surgeries = df['Surgery_Type'].value_counts().head(5).index
plt.figure(figsize=(10,6))
sns.boxplot(data=df[df['Surgery_Type'].isin(top_surgeries)], x='Surgery_Type', y='Treatment_Days', palette='Set2')
plt.title('Treatment Duration by Surgery Type')
plt.xlabel('Surgery Type')
plt.ylabel('Treatment Days')
plt.xticks(rotation=45)
plt.show()

# Treatment vs Recovery Correlation
df_corr = df[['Treatment_Days', 'Recovery_Days']].dropna()
corr, p_val = pearsonr(df_corr['Treatment_Days'], df_corr['Recovery_Days'])
print(f"\nCorrelation between Treatment and Recovery Days: r = {corr:.3f}, p = {p_val:.4f}")

sns.lmplot(data=df_corr, x='Treatment_Days', y='Recovery_Days', aspect=1.5)
plt.title('Relationship: Treatment vs Recovery Duration')
plt.xlabel('Treatment Days')
plt.ylabel('Recovery Days')
plt.show()

# 3. Does Hospital Affect Recovery Time for the Most Common Diagnosis?
top_diagnosis = df['Diagnosis'].value_counts().idxmax()
print(f"\nMost Common Diagnosis: {top_diagnosis}")
df_diag_hosp = df[df['Diagnosis'] == top_diagnosis]
top_hospitals = df_diag_hosp['Hospital_Name'].value_counts().head(5).index

plt.figure(figsize=(12,6))
sns.boxplot(data=df_diag_hosp[df_diag_hosp['Hospital_Name'].isin(top_hospitals)],
            x='Hospital_Name', y='Recovery_Days', palette='Set3')
plt.title(f'Recovery Days Across Hospitals for: {top_diagnosis}')
plt.xlabel('Hospital Name')
plt.ylabel('Recovery Days')
plt.xticks(rotation=45)
plt.show()

# 4. Are Younger Patients Recovering Faster?
df_age_rec = df[['Age', 'Recovery_Days']].dropna()
corr_age, p_age = pearsonr(df_age_rec['Age'], df_age_rec['Recovery_Days'])
print(f"\nCorrelation between Age and Recovery Days: r = {corr_age:.3f}, p = {p_age:.4f}")

sns.lmplot(data=df_age_rec, x='Age', y='Recovery_Days', height=5, aspect=1.5)
plt.title('Does Age Impact Recovery Time?')
plt.xlabel('Age')
plt.ylabel('Recovery Days')
plt.show()

# 5. Is There a Gender Difference in Recovery Time?
gender_crosstab = pd.crosstab(df['Gender'], df['Diagnosis'])
print("\nGender Distribution Across Diagnoses:")
print(gender_crosstab)

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Gender', y='Recovery_Days', palette='coolwarm')
plt.title('Recovery Days by Gender')
plt.xlabel('Gender')
plt.ylabel('Recovery Days')
plt.show()


# SENTIMENT ANALYSIS
df['Polarity'] = df['Feedback'].apply(lambda text: TextBlob(str(text)).sentiment.polarity)
df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral')

# Count and percentage
sentiment_counts = df['Sentiment'].value_counts(normalize=True).mul(100).round(2)  # % values
sentiment_order = ['Positive', 'Neutral', 'Negative']  # Ensures consistent order

# Plot
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Sentiment', order=sentiment_order)

# Annotate with percentage
for p in ax.patches:
    count = p.get_height()
    sentiment = p.get_x() + p.get_width() / 2
    percent = (count / len(df)) * 100
    ax.annotate(f'{percent:.1f}%', (p.get_x() + p.get_width() / 2, count),
                ha='center', va='bottom', fontsize=12)

# Labels and styling
plt.title('Sentiment Distribution of Patient Feedback')
plt.xlabel('Sentiment')
plt.ylabel('Number of Feedbacks')
plt.tight_layout()
plt.show()

# --- Feature Selection & Target Variables ---
features = ['Age', 'Gender', 'Heart_Rate', 'Temperature',
            'Systolic', 'Diastolic', 'X-ray_Results',
            'Lab_Test_Results', 'FamilyHistory', 'Allergies', 'Hypertension_Risk']

X = df[features].copy()
y_cls = df['Diagnosis']
y_reg = df['Recovery_Days']

# Encode Lab test if needed
if X['Lab_Test_Results'].dtype == 'object':
    le_lab = LabelEncoder()
    X['Lab_Test_Results'] = le_lab.fit_transform(X['Lab_Test_Results'])
    joblib.dump(le_lab, 'le_lab.pkl')

# Remove rare classes
valid_classes = y_cls.value_counts()[y_cls.value_counts() >= 2].index
mask = y_cls.isin(valid_classes)
X = X[mask]
y_cls = y_cls[mask]
y_reg = y_reg[mask]

X = X[mask]
y_cls = y_cls[mask]
y_reg = y_reg[mask]

# Drop rows or fill missing
X = X.fillna(X.median(numeric_only=True))

# Remove constant columns
X = X.loc[:, X.nunique() > 1]

# Storing features list
joblib.dump(X.columns.tolist(), 'model_features.pkl')

# Final check
if X.isnull().values.any():
    raise ValueError("Still NaNs in X")


# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

if X.isnull().values.any():
    print("NaNs still exist in X!")
    print(X.isnull().sum())
    raise ValueError("Clean your dataset: NaNs found in X before model fitting.")

# --- Classification: Predicting Diagnosis ---

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X_scaled, y_cls):
    X_train_cls, X_test_cls = X_scaled[train_idx], X_scaled[test_idx]
    y_train_cls, y_test_cls = y_cls.iloc[train_idx], y_cls.iloc[test_idx]

class_models = {
    'RandomForest': RandomForestClassifier(),
    'HistGradientBoosting': HistGradientBoostingClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=500),
    'SVM': SVC()
}

print("\nDiagnosis Prediction - Classification Results:\n")
best_acc, best_cls_model = 0, None
for name, model in class_models.items():
    model.fit(X_train_cls, y_train_cls)
    preds = model.predict(X_test_cls)
    acc = accuracy_score(y_test_cls, preds)
    print(f"{name}: Accuracy = {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_cls_model = model

joblib.dump(best_cls_model, 'diagnosis_model.pkl')

# --- Regression: Predicting Recovery Days ---

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42)

reg_models = {
    'RandomForestRegressor': RandomForestRegressor(),
    'XGBoostRegressor': XGBRegressor(),
    'LinearRegression': LinearRegression(),
    'GradientBoosting': GradientBoostingRegressor(),
    'SVR': SVR()
}

print("\nRecovery Time Prediction - Regression Results:\n")
best_r2, best_reg_model = -1, None
for name, model in reg_models.items():
    model.fit(X_train_reg, y_train_reg)
    preds = model.predict(X_test_reg)
    r2 = r2_score(y_test_reg, preds)
    mae = mean_absolute_error(y_test_reg, preds)
    print(f"{name}: R2 = {r2:.4f}, MAE = {mae:.2f}")
    if r2 > best_r2:
        best_r2 = r2
        best_reg_model = model

joblib.dump(best_reg_model, 'recovery_model.pkl')

# --- Scenario-Based Prediction (Optional Testing in Console) ---

# Load the models
clf_model = joblib.load('diagnosis_model.pkl')
reg_model = joblib.load('recovery_model.pkl')
scaler = joblib.load('scaler.pkl')

# Save metrics
joblib.dump(best_acc, 'classification_accuracy.pkl')
joblib.dump(best_r2, 'regression_r2.pkl')

# Define consistent input order
feature_cols = features

# Scenario 1: Mr. Harry
harry_input = {
    'Age': 70,
    'Gender': 0,
    'Heart_Rate': 60,
    'Temperature': 97,
    'Systolic': 120,
    'Diastolic': 80,
    'X-ray_Results': 1,
    'Lab_Test_Results': 83,
    'FamilyHistory': 1,
    'Allergies': 0,
    'Hypertension_Risk': int(120 >= 130 or 80 >= 80)
}

harry_df = pd.DataFrame([harry_input])[X.columns]
harry_scaled = scaler.transform(harry_df)
harry_pred = clf_model.predict(harry_scaled)[0]
print(f"\n--- Scenario 1: Diagnosis for Mr. Harry: {harry_pred}")

# Scenario 2: Ms. Reena
reena_input = {
    'Age': 40,
    'Gender': 1,
    'Heart_Rate': 75,
    'Temperature': 98.6,
    'Systolic': 120,
    'Diastolic': 75,
    'X-ray_Results': 0,
    'Lab_Test_Results': 90,
    'FamilyHistory': 0,
    'Allergies': 1,
    'Hypertension_Risk': int(120 >= 130 or 80 >= 80)
}

reena_df = pd.DataFrame([reena_input])[X.columns]
reena_scaled = scaler.transform(reena_df)
reena_pred_days = reg_model.predict(reena_scaled)[0]
print(f"--- Scenario 2: Recovery Time for Ms. Reena: {round(reena_pred_days, 1)} days")
