# Heart Disease Risk Analysis and Prediction

## Project Overview

This project analyzes and predicts heart disease risk using machine learning techniques applied to the UCI Heart Disease dataset. The workflow covers data preprocessing, exploratory data analysis, dimensionality reduction via PCA, feature selection, supervised classification, unsupervised clustering, and hyperparameter tuning.

## Objectives

- Clean and preprocess raw heart disease data, handling missing values and encoding categorical variables
- Reduce dimensionality with Principal Component Analysis (PCA)
- Rank features using Random Forest, Gradient Boosting, and Chi-Square methods
- Train and evaluate multiple classification models
- Discover natural patient groupings using K-Means and Hierarchical Clustering
- Optimize model performance through GridSearchCV hyperparameter tuning

## Technologies Used

- Python 3.x
- Pandas — data manipulation and analysis
- NumPy — numerical computing
- Scikit-learn — machine learning algorithms, preprocessing, and evaluation
- SciPy — hierarchical clustering
- Matplotlib — data visualization
- Seaborn — statistical data visualization
- ucimlrepo — UCI ML Repository dataset access
- Joblib — model serialization

## Dataset Information

- **Source**: UCI Machine Learning Repository (Heart Disease Dataset, ID: 45)
- **Raw dataset**: 303 rows, 14 columns
- **Cleaned dataset**: 297 rows, 25 feature columns (after dropping nulls and one-hot encoding)
- **Target**: Binary classification — 0 = no significant narrowing (< 50%), 1 = significant narrowing (> 50%)

### Dataset Attributes

| # | Attribute | Description | Values / Range |
|---|-----------|-------------|----------------|
| 1 | age | Patient age | Integer |
| 2 | sex | Gender | 1 = male; 0 = female |
| 3 | cp | Chest pain type | 1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic |
| 4 | trestbps | Resting blood pressure | mm Hg on admission |
| 5 | chol | Serum cholesterol | mg/dl |
| 6 | fbs | Fasting blood sugar > 120 mg/dl | 1 = true; 0 = false |
| 7 | restecg | Resting ECG results | 0 = normal; 1 = ST-T wave abnormality; 2 = left ventricular hypertrophy |
| 8 | thalach | Maximum heart rate achieved | Integer |
| 9 | exang | Exercise-induced angina | 1 = yes; 0 = no |
| 10 | oldpeak | ST depression induced by exercise relative to rest | Float |
| 11 | slope | Slope of peak exercise ST segment | 1 = upsloping; 2 = flat; 3 = downsloping |
| 12 | ca | Number of major vessels colored by fluoroscopy | 0–3 |
| 13 | thal | Thalassemia | 3 = normal; 6 = fixed defect; 7 = reversible defect |
| 14 | target | Heart disease diagnosis (target variable) | 0 = absent; 1 = present |

## Project Structure

```
Heart_Disease_Project/
|
|-- data/
|   |-- heart_disease.csv          # Raw dataset fetched from UCI
|   |-- cleaned_df.csv             # Cleaned and encoded full DataFrame
|   |-- cleaned_X.csv              # Cleaned feature matrix (25 columns)
|   `-- clean_y.csv                # Cleaned target vector
|
|-- notebooks/
|   |-- 01_data_preprocessing.ipynb
|   |-- 02_pca_analysis.ipynb
|   |-- 03_feature_selection.ipynb
|   |-- 04_supervised_learning.ipynb
|   |-- 05_unsupervised_learning.ipynb
|   `-- 06_hyperparameter_tuning.ipynb
|
|-- models/
|   `-- final_model.pkl            # Best performing model (SVM, saved with joblib)
|
|-- results/
|   `-- evaluation_metrics.txt     # Model performance metrics
|
|-- README.md
`-- requirements.txt
```

## Installation

```bash
git clone [repository-url]
cd Heart_Disease_Project
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
ucimlrepo>=0.0.3
joblib>=1.1.0
jupyter>=1.0.0
```

## Workflow

### 1. Data Preprocessing (`01_data_preprocessing.ipynb`)

The dataset is fetched directly from the UCI ML Repository using the `ucimlrepo` library (dataset ID 45) and saved locally as `heart_disease.csv`. The preprocessing steps are as follows:

**Missing value handling:** Six rows containing null values were identified and removed, reducing the dataset from 303 to 297 samples.

**One-hot encoding:** Seven categorical columns were encoded into binary dummy variables — `sex`, `cp`, `restecg`, `slope`, `fbs`, `exang`, and `thal` — expanding the feature space from 13 to 25 columns.

**Exploratory Data Analysis:** An age distribution histogram, a sex distribution pie chart, per-feature histograms with KDE curves, and a correlation heatmap were produced to understand data distributions and inter-feature relationships.

![Correlation Matrix](results/charts/chart2_correlation.png)

**Output files:** Three CSV files are saved for downstream use — `cleaned_df.csv` (full encoded DataFrame), `cleaned_X.csv` (feature matrix only), and `clean_y.csv` (target vector).

![Exploratory Data Analysis](results/charts/chart1_eda.png)

### 2. PCA Analysis (`02_pca_analysis.ipynb`)

PCA is applied after StandardScaler normalization. The number of components is chosen to retain 95% of total variance (`n_components=0.95`). Two visualizations are produced: a 2D scatter plot of the first two principal components colored by target label, and a cumulative explained variance curve showing the trade-off between the number of components and variance retained.

![PCA Analysis](results/charts/chart4_pca.png)

### 3. Feature Selection (`03_feature_selection.ipynb`)

Three complementary feature selection approaches are applied to the cleaned data after an 80/20 train-test split and standard scaling.

**Random Forest importance:** A RandomForestClassifier is trained and feature importances are extracted and ranked. The top five features are `ca` (0.143), `thalach` (0.104), `age` (0.090), `oldpeak` (0.086), and `chol` (0.078).

**Gradient Boosting importance:** A GradientBoostingClassifier is trained in the same way, producing the same top five features with identical ranking, confirming the robustness of these importance scores across ensemble methods.

**Chi-Square selection:** `SelectKBest` with `chi2` is applied to the full (unscaled) feature set. The top five features by chi-square score are `thalach` (187.05), `ca` (82.73), `oldpeak` (68.57), `thal_7.0` (42.75), and `cp_4` (39.85). Results are consistent with ensemble-based rankings, with `thalach` and `ca` appearing prominently across all three methods.

![Feature Selection](results/charts/chart3_feature_selection.png)

### 4. Supervised Learning (`04_supervised_learning.ipynb`)

All models are trained on the same 80/20 split with StandardScaler normalization applied to both train and test sets. A scikit-learn `Pipeline` is also used to train all four models in parallel for comparison. The best model (SVM) is saved to `models/final_model.pkl` using joblib.

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 0.8833 | Linear classifier with probabilistic output |
| Decision Tree | 0.8167 | `max_depth=5` to control overfitting |
| Random Forest | 0.8700 | Ensemble of decision trees, default parameters |
| SVM | 0.9000 | RBF kernel, `random_state=42` |

In addition to accuracy, confusion matrix heatmaps and full classification reports (precision, recall, F1-score) are produced for each model. A decision tree visualization and Random Forest / Gradient Boosting feature importance bar charts are also generated with annotated importance values.

![Model Performance](results/charts/chart5_model_performance.png)

### 5. Unsupervised Learning (`05_unsupervised_learning.ipynb`)

**K-Means Clustering:** The elbow method is used to evaluate WCSS (Within-Cluster Sum of Squares) for K values from 1 to 10. Based on the elbow plot, K=4 is selected as the optimal number of clusters. A 2D scatter plot displays cluster assignments against the first two scaled features, with centroids marked in red.

**Hierarchical Clustering:** Ward linkage is applied to the scaled feature matrix and visualized as a dendrogram. The dendrogram reveals the hierarchical structure of patient groupings and helps corroborate the cluster count suggested by the elbow method.

![Clustering Analysis](results/charts/chart6_clustering.png)

### 6. Hyperparameter Tuning (`06_hyperparameter_tuning.ipynb`)

GridSearchCV with 5-fold cross-validation is used to tune the Logistic Regression model. The parameter grid covers regularization strengths `C` in `[0.01, 0.1, 1, 10]` and solvers `liblinear` and `lbfgs`.

The best parameters found are `C=1` with `solver='liblinear'`, producing a test accuracy of 0.8833 — identical to the baseline, confirming the default parameters were already near-optimal for this dataset.

## Results Summary

| Model | Accuracy | Class 0 F1 | Class 1 F1 |
|-------|----------|------------|------------|
| Logistic Regression | 0.8833 | 0.90 | 0.86 |
| Decision Tree | 0.8167 | 0.85 | 0.78 |
| Random Forest | 0.8700 | 0.89 | 0.83 |
| SVM (best model) | 0.9000 | 0.92 | 0.87 |

The SVM with RBF kernel achieves the highest accuracy (90%) and is saved as the final model. The most predictive features across all selection methods are `thalach` (maximum heart rate), `ca` (number of major vessels), and `oldpeak` (ST depression).

## Usage

Run notebooks in sequence:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_pca_analysis.ipynb
jupyter notebook notebooks/03_feature_selection.ipynb
jupyter notebook notebooks/04_supervised_learning.ipynb
jupyter notebook notebooks/05_unsupervised_learning.ipynb
jupyter notebook notebooks/06_hyperparameter_tuning.ipynb
```

The final trained model is saved to `models/final_model.pkl` and evaluation metrics are stored in `results/evaluation_metrics.txt`.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## Contact

Mena Beshara
https://uk.linkedin.com/in/menabeshara
