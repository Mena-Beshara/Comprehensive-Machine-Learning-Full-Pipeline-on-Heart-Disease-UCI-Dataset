# Heart Disease Risk Analysis and Prediction

## 📋 Project Overview

This project aims to analyze, predict, and visualize heart disease risks using machine learning techniques. The comprehensive workflow involves data preprocessing, feature selection, dimensionality reduction (PCA), model training, evaluation, and deployment.

## 🎯 Objectives

- **Data Preprocessing & Cleaning**: Handle missing values, encoding, and scaling
- **Dimensionality Reduction**: Apply PCA to retain essential features
- **Feature Selection**: Use statistical methods and ML-based techniques
- **Supervised Learning**: Train classification models (Logistic Regression, Decision Trees, Random Forest, SVM)
- **Unsupervised Learning**: Apply K-Means and Hierarchical Clustering for pattern discovery
- **Model Optimization**: Implement hyperparameter tuning using GridSearchCV and RandomizedSearchCV

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **ucimlrepo** - UCI ML Repository dataset access

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository (Heart Disease Dataset)
- **Dataset ID**: 45
- **Type**: Classification dataset for heart disease prediction
- **Features**: 14 medical attributes for heart disease prediction
- **Target**: Binary classification (presence/absence of heart disease)

### Dataset Attributes

| # | Attribute | Description | Values/Range |
|---|-----------|-------------|--------------|
| 1 | age | Age of the patient | Numerical |
| 2 | sex | Gender | 1 = male; 0 = female |
| 3 | cp | Chest pain type | 1 = typical angina<br>2 = atypical angina<br>3 = non-anginal pain<br>4 = asymptomatic |
| 4 | trestbps | Resting blood pressure | mm Hg on admission |
| 5 | chol | Serum cholesterol | mg/dl |
| 6 | fbs | Fasting blood sugar > 120 mg/dl | 1 = true; 0 = false |
| 7 | restecg | Resting electrocardiographic results | 0 = normal<br>1 = ST-T wave abnormality<br>2 = left ventricular hypertrophy |
| 8 | thalach | Maximum heart rate achieved | Numerical |
| 9 | exang | Exercise induced angina | 1 = yes; 0 = no |
| 10 | oldpeak | ST depression induced by exercise | Relative to rest |
| 11 | slope | Slope of peak exercise ST segment | 1 = upsloping<br>2 = flat<br>3 = downsloping |
| 12 | ca | Number of major vessels colored by fluoroscopy | 0-3 |
| 13 | thal | Thalassemia | 3 = normal<br>6 = fixed defect<br>7 = reversible defect |
| 14 | num | **Target Variable**: Heart disease diagnosis | 0 = < 50% diameter narrowing<br>1 = > 50% diameter narrowing |

## 🔧 Installation

```bash
# Clone the repository
git clone [repository-url]
cd Heart_Disease_Project

# Install required packages
pip install -r requirements.txt
```

### Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
ucimlrepo>=0.0.3
jupyter>=1.0.0
```

## 🚀 Getting Started

### Data Loading
```python
# Import required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Fetch the Heart Disease dataset from UCI ML Repository
heart_disease = fetch_ucirepo(id=45)

# Extract features and target variables
X = heart_disease.data.features  # Feature matrix
y = heart_disease.data.targets   # Target variable

# Display dataset metadata and variable information
print(heart_disease.metadata)
print(heart_disease.variables)
```

## 📁 Project Structure

```
Heart_Disease_Project/
│
├── data/
│   └── heart_disease.csv              # Heart disease dataset
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Data cleaning and preprocessing
│   ├── 02_pca_analysis.ipynb          # Principal Component Analysis
│   ├── 03_feature_selection.ipynb    # Feature selection techniques
│   ├── 04_supervised_learning.ipynb  # Classification models
│   ├── 05_unsupervised_learning.ipynb # Clustering analysis
│   └── 06_hyperparameter_tuning.ipynb # Model optimization
│
├── models/
│   └── final_model.pkl               # Trained and optimized model
│
├── results/
│   └── evaluation_metrics.txt        # Model performance metrics
│
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

## 🚀 Workflow

### 1. Data Loading and Initial Setup
- Load the Heart Disease dataset from UCI ML Repository using `ucimlrepo`
- Extract feature matrix (X) and target variable (y)
- Display dataset metadata and variable information for initial understanding

### 2. Data Preprocessing (`01_data_preprocessing.ipynb`)

#### Data Loading and Initial Exploration
```python
# Load the heart disease dataset
df = pd.read_csv('../data/heart_disease.csv')

# Display basic information about the dataset
print('\nDataFrame info:')
df.info()

# Generate summary statistics for numerical features
print(f'\nSummary Statistics: \n: {df.describe()}')

# Handle missing values (if any)
df_dropped = df.dropna()  # Remove rows with missing values

# Separate features and target variable
X = df.drop('target', axis=1)  # Feature matrix
y = df['target']               # Target variable (heart disease diagnosis)
```

#### Categorical Variable Encoding
```python
# Apply one-hot encoding to categorical variables
df_encoded = pd.get_dummies(df_dropped, columns=['sex', 'cp', 'restecg', 'slope', 'fbs', 'exang', 'thal'])
```

**Key Steps:**
- Load dataset from CSV file
- Explore data structure and types using `df.info()`
- Generate descriptive statistics for all numerical features
- Remove any missing values to ensure clean data
- **One-Hot Encoding**: Convert categorical variables into binary dummy variables
  - `sex`: Gender (male/female)
  - `cp`: Chest pain type (4 categories)
  - `restecg`: Resting ECG results (3 categories)
  - `slope`: ST segment slope (3 categories)
  - `fbs`: Fasting blood sugar (binary)
  - `exang`: Exercise induced angina (binary)
  - `thal`: Thalassemia type (3 categories)
#### Data Splitting and Scaling
```python
# Import required libraries for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split dataset into training and testing sets
# Test size = 20%, Train size = 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)        # Transform test data using fitted scaler
```

### 3. PCA Analysis (`02_pca_analysis.ipynb`)
*[Details to be added based on your PCA implementation]*

### 4. Feature Selection (`03_feature_selection.ipynb`)

#### Chi-Square Statistical Feature Selection
```python
from sklearn.feature_selection import SelectKBest, chi2

# Apply Chi-square feature selection 
chi2_selector = SelectKBest(chi2, k='all')
X_new = chi2_selector.fit_transform(X, y)

# Get the scores for each feature
feature_scores = chi2_selector.scores_

# Create a DataFrame to view the feature scores 
features_scores_df = pd.DataFrame({
    'Feature': X.columns,
    'Chi-Square Score': feature_scores
})

# Sort features based on Chi-square score
feature_scores_df = features_scores_df.sort_values(by='Chi-Square Score', ascending=False)

feature_scores_df
```

**Key Steps:**
- **Statistical Feature Selection**: Chi-square test to measure dependency between features and target
- **Comprehensive Analysis**: Test all features (`k='all'`) to get complete ranking
- **Feature Scoring**: Calculate chi-square scores for each medical attribute
- **Ranked Results**: Sort features by statistical significance for heart disease prediction

### 5. Supervised Learning (`04_supervised_learning.ipynb`)

#### Logistic Regression Model
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create and train Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

# Make Predictions 
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate Model 
print('Logistic Regression Model Results:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}')
print('\nClassification Report: ')
print(classification_report(y_test, y_pred_lr))

# Plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix Using Logistic Regression Model')
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.show()
```

#### Decision Tree Model
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create and train Decision Tree Model 
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred_dt = dt_model.predict(X_test_scaled)

# Evaluate Model 
print('Decision Tree Model Results:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred_dt))

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns.tolist(), 
          class_names=[str(i) for i in range(len(y))])
plt.title('Decision Tree Visualization')
plt.show()
```

#### Random Forest Model
```python
from sklearn.ensemble import RandomForestClassifier

# Create and train Random Forest Model 
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Make predictions 
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate Model 
print('Random Forest Results:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred_rf))
```

#### Feature Importance Analysis
```python
# Plot Feature Importances using Random Forest
# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Get feature importances and create DataFrame for easier handling
importances_rf = rf_model.feature_importances_
feature_names = X.columns.tolist()

# Create a DataFrame and sort by importance (descending)
importance_rf = pd.DataFrame({
    'feature': feature_names,
    'importance': importances_rf
}).sort_values('importance', ascending=False)

# Create the plot
plt.figure(figsize=(12, 8))

# Create vertical bar plot with seaborn color palette
ax = sns.barplot(data=importance_rf, x='feature', y='importance', 
                 palette='viridis', hue='feature', legend=False)

# Customize the plot
plt.title('Feature Importances - Random Forest', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Importance Score', fontsize=12, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on top of bars
for i, (bar, importance) in enumerate(zip(ax.patches, importance_rf['importance'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
            f'{importance:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Customize grid
plt.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Remove top and right spines
sns.despine()

# Adjust layout
plt.tight_layout()
plt.show()
```

#### Gradient Boosting Model
```python
from sklearn.ensemble import GradientBoostingClassifier

# Create and train Gradient Boosting Model 
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred_gb = gb_model.predict(X_test_scaled)
```

#### Support Vector Machine (SVM) Model
```python
from sklearn.svm import SVC

# Create and train SVM Model 
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate Model 
print('SVM Model Results:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}')
print('\nClassification Report: ')
print(classification_report(y_test, y_pred_svm))
```

**Model Implementation Features:**
- **Logistic Regression**: Linear classifier with probabilistic output
- **Decision Tree**: Tree-based classifier with interpretable rules (max_depth=5 for complexity control)
- **Random Forest**: Ensemble method combining multiple decision trees for improved accuracy and reduced overfitting
- **Gradient Boosting**: Sequential ensemble method building models iteratively to correct previous errors
- **Support Vector Machine**: Non-linear classifier using RBF (Radial Basis Function) kernel for complex decision boundaries
- **Feature Importance Analysis**: Both Random Forest and Gradient Boosting-based feature ranking with comprehensive visualization
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score via classification reports
- **Visual Analysis**: 
  - Confusion matrix heatmaps for prediction analysis
  - Decision tree visualization showing feature splits and decision paths
  - Feature importance bar charts for both Random Forest and Gradient Boosting models
  - Professional styling with value annotations and custom aesthetics
- **Consistent Evaluation**: Same metrics applied across all models for comparison

### 6. Unsupervised Learning (`05_unsupervised_learning.ipynb`)

#### K-Means Clustering with Elbow Method
```python
from sklearn.cluster import KMeans

# Elbow Method to determine optimal K
# Range of K values to test
k_range = range(1, 11)  # Testing K from 1 to 10
wcss = []  # Within-Cluster Sum of Squares

# Calculate WCSS for each K
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ gives us the WCSS

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

# Based on the elbow plot, choose optimal K = 4
optimal_k = 4

# Perform K-Means with optimal K
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=0)
kmeans_optimal.fit(X_scaled)

# Plot Results with optimal K
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_optimal.labels_, cmap='viridis')
plt.scatter(kmeans_optimal.cluster_centers_[:, 0], kmeans_optimal.cluster_centers_[:, 1], 
           marker='x', s=200, linewidth=3, color='r')
plt.title(f'K-Means Clustering with Optimal K={optimal_k}')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.show()
```

**Key Steps:**
- **Elbow Method**: Systematic approach to determine optimal number of clusters (K=1 to 10)
- **WCSS Analysis**: Calculate Within-Cluster Sum of Squares for each K value
- **Optimal K Selection**: Choose K=4 based on elbow point in the WCSS plot
- **Cluster Visualization**: 2D scatter plot showing cluster assignments and centroids
- **Pattern Discovery**: Identify natural groupings in heart disease patient data



## 📈 Results
Logistic Regression Model Results:
Accuracy: 0.8833

Classification Report: 
              precision    recall  f1-score   support

           0       0.91      0.89      0.90        36
           1       0.84      0.88      0.86        24


Decison Tree Model Results:
Accuracy: 0.8167

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.83      0.85        36
           1       0.76      0.79      0.78        24

Accuracy: 0.87

Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.86      0.89        36
           1       0.81      0.88      0.84        24


SVM Model Results:
Accuracy: 0.900

Classification Report: 
              precision    recall  f1-score   support

           0       0.89      0.94      0.92        36
           1       0.91      0.83      0.87        24

## 🔍 Usage

1. **Start with Data Preprocessing:**
   ```bash
   jupyter notebook notebooks/01_data_preprocessing.ipynb
   ```

2. **Run PCA Analysis:**
   ```bash
   jupyter notebook notebooks/02_pca_analysis.ipynb
   ```

3. **Perform Feature Selection:**
   ```bash
   jupyter notebook notebooks/03_feature_selection.ipynb
   ```

4. **Train Supervised Learning Models:**
   ```bash
   jupyter notebook notebooks/04_supervised_learning.ipynb
   ```

5. **Apply Unsupervised Learning:**
   ```bash
   jupyter notebook notebooks/05_unsupervised_learning.ipynb
   ```

6. **Optimize Models:**
   ```bash
   jupyter notebook notebooks/06_hyperparameter_tuning.ipynb
   ```

The final optimized model will be saved as `models/final_model.pkl` and evaluation metrics will be stored in `results/evaluation_metrics.txt`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📧 Contact

Mena Beshara
https://uk.linkedin.com/in/menabeshara

**Note**: This README will be updated with detailed technical information as the code components are shared.