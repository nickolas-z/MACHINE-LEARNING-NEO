# Імпорти
import os
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

warnings.filterwarnings('ignore')

# Create a folder for saving plots if it doesn't exist
output_dir = "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/output"
os.makedirs(output_dir, exist_ok=True)

# Initialize these variables in global scope to avoid reference errors in except blocks
train_df = None
test_df = None
validation_df = None
X = None
y = None
num_features = []
cat_features = []

try:
    # 1. Завантаження даних
    print("Step 1: Loading and initial examination of the data")
    train_df = pd.read_csv(
        "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/datasets/final/final_proj_data.csv"
    )
    test_df = pd.read_csv(
        "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/datasets/final/final_proj_test.csv"
    )
    
    # Try to load validation file but handle if it doesn't exist
    validation_path = "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/datasets/final/final_proj_validation.csv"
    try:
        validation_df = pd.read_csv(validation_path)
        has_validation = True
        print("Validation data loaded successfully")
    except FileNotFoundError:
        print(f"Validation file not found at {validation_path}")
        print("Will create a subset of test data for validation purposes")
        # Split test data to create a validation set
        test_df, validation_df = train_test_split(test_df, test_size=0.3, random_state=42)
        has_validation = True

    # 2. Початковий аналіз даних
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    if has_validation:
        print(f"Validation data shape: {validation_df.shape}")
    
    # Analyze data types
    print("\nData types in training set:")
    print(train_df.dtypes.value_counts())
    
    # Analyze missing values
    missing_summary = train_df.isnull().sum()
    missing_percent = (missing_summary / len(train_df)) * 100
    missing_data = pd.DataFrame({'Missing Count': missing_summary, 
                                'Missing Percent': missing_percent})
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Percent', ascending=False)
    
    print("\nFeatures with missing values:")
    print(missing_data)
    
    # 2.1 Remove completely empty columns
    print("\nStep 2: Cleaning and preprocessing data")
    columns_to_drop = missing_summary[missing_summary == len(train_df)].index.tolist()
    if columns_to_drop:
        print(f"Removing {len(columns_to_drop)} completely empty columns: {columns_to_drop}")
        train_df.drop(columns=columns_to_drop, inplace=True)
        test_df.drop(columns=columns_to_drop, errors="ignore", inplace=True)
        if has_validation:
            validation_df.drop(columns=columns_to_drop, errors="ignore", inplace=True)
    
    # 3. Ensure dataset compatibility
    # Make sure test and validation sets have the same columns as train set
    for col in train_df.columns:
        if col != 'y' and col not in test_df.columns:
            test_df[col] = np.nan
        if has_validation and col != 'y' and col not in validation_df.columns:
            validation_df[col] = np.nan
    
    # 4. EDA - Target variable analysis
    plt.figure(figsize=(10, 6))
    target_counts = train_df['y'].value_counts()
    ax = sns.countplot(x="y", data=train_df)
    plt.title("Distribution of Target Variable (y)")
    
    # Add percentages on top of bars
    for i, count in enumerate(target_counts):
        percentage = 100 * count / len(train_df)
        ax.text(i, count + 5, f"{percentage:.1f}%", ha='center')
    
    plt.savefig(f"{output_dir}/target_distribution.png")
    plt.show()
    
    class_imbalance = train_df['y'].value_counts().max() / train_df['y'].value_counts().min()
    print(f"\nClass imbalance ratio (majority:minority): {class_imbalance:.2f}")
    
    # 5. Missing values visualization
    plt.figure(figsize=(14, 8))
    sns.heatmap(train_df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Missing Values in Training Dataset")
    plt.savefig(f"{output_dir}/missing_values_heatmap.png")
    plt.show()
    
    # 6. Feature Analysis
    # 6.1 Separate features by type
    X = train_df.drop(columns=["y"], errors="ignore")
    y = train_df["y"]
    num_features = X.select_dtypes(include="number").columns.tolist()
    cat_features = X.select_dtypes(include="object").columns.tolist()
    
    print(f"\nNumeric features: {len(num_features)}")
    print(f"Categorical features: {len(cat_features)}")
    
    # 6.2 Analyze categorical features
    if cat_features:
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(len(cat_features) // 2 + 1, 2)
        
        for i, feature in enumerate(cat_features):
            ax = plt.subplot(gs[i // 2, i % 2])
            
            # Count unique values excluding NaN
            unique_count = train_df[feature].dropna().nunique()
            value_counts = train_df[feature].value_counts(dropna=False).head(10)
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f"{feature} (unique: {unique_count})")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
        
        plt.savefig(f"{output_dir}/categorical_features.png")
        plt.show()
        
        # Print cardinality of categorical features
        cat_cardinality = {col: train_df[col].nunique() for col in cat_features}
        sorted_cardinality = {k: v for k, v in sorted(cat_cardinality.items(), key=lambda item: item[1], reverse=True)}
        print("\nCardinality of categorical features:")
        for feature, cardinality in sorted_cardinality.items():
            print(f"{feature}: {cardinality} unique values")

    # 6.3 Analyze numeric features
    if num_features:
        # Distribution of numeric features
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(min(10, len(num_features)) // 2 + 1, 2)
        
        for i, feature in enumerate(num_features[:10]):  # Limit to first 10 features for visibility
            ax = plt.subplot(gs[i // 2, i % 2])
            sns.histplot(train_df[feature].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            plt.tight_layout()
        
        plt.savefig(f"{output_dir}/numeric_distributions.png")
        plt.show()
        
        # Correlation matrix
        plt.figure(figsize=(14, 10))
        numeric_data = train_df[num_features].copy()
        corr_matrix = numeric_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, center=0)
        plt.title("Correlation Matrix of Numeric Features")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.show()
        
        # Find highly correlated features
        high_corr_threshold = 0.8
        high_corr_features = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
                    feat_pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                    high_corr_features.append(feat_pair)
        
        if high_corr_features:
            print("\nHighly correlated feature pairs:")
            for feat1, feat2, corr in high_corr_features:
                print(f"{feat1} and {feat2}: {corr:.2f}")
    
    # 7. Preliminary feature importance analysis
    # Using a simple Random Forest to understand feature importance before preprocessing
    X_sample = X.copy()
    
    # Temporarily fill missing values for feature importance analysis
    for col in num_features:
        X_sample[col] = X_sample[col].fillna(X_sample[col].median())
    for col in cat_features:
        X_sample[col] = X_sample[col].fillna(X_sample[col].mode()[0] if not X_sample[col].mode().empty else "Unknown")
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X_sample, columns=cat_features, drop_first=True)
    
    # Run a quick Random Forest to get feature importance
    print("\nRunning preliminary feature importance analysis...")
    rf_prelim = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_prelim.fit(X_encoded, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': rf_prelim.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Features by Importance (Preliminary - Random Forest)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/preliminary_feature_importance_rf.png")
    plt.show()
    
    # PCA-based feature importance analysis
    print("\nRunning PCA-based feature importance analysis...")
    # Scale data for PCA
    X_scaled = StandardScaler().fit_transform(X_encoded)
    
    # Apply PCA
    pca = PCA(n_components=min(20, X_scaled.shape[1]))
    pca.fit(X_scaled)
    
    # Calculate feature importance based on PCA loadings
    # We'll use the sum of absolute loadings weighted by explained variance ratio
    loadings = pca.components_.T  # Transpose to get features in rows
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Calculate weighted importance scores
    pca_importance = np.zeros(loadings.shape[0])
    for i in range(len(explained_variance_ratio)):
        pca_importance += np.abs(loadings[:, i]) * explained_variance_ratio[i]
    
    # Normalize to get relative importance
    pca_importance = pca_importance / np.sum(pca_importance)
    
    # Create DataFrame with feature names and importance scores
    pca_feature_importance = pd.DataFrame({
        'Feature': X_encoded.columns,
        'PCA_Importance': pca_importance
    }).sort_values('PCA_Importance', ascending=False)
    
    # Plot PCA-based feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='PCA_Importance', y='Feature', data=pca_feature_importance.head(20))
    plt.title('Top 20 Features by Importance (PCA-based Analysis)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_pca.png")
    plt.show()
    
    # Compare feature importance rankings from Random Forest and PCA
    # Get top 20 features from each method
    rf_top_features = feature_importance.head(20)['Feature'].tolist()
    pca_top_features = pca_feature_importance.head(20)['Feature'].tolist()
    
    # Find common features
    common_features = set(rf_top_features).intersection(set(pca_top_features))
    
    print(f"\nNumber of common features in top 20 between RF and PCA: {len(common_features)}")
    print("Common important features:")
    for feature in common_features:
        rf_rank = rf_top_features.index(feature) + 1
        pca_rank = pca_top_features.index(feature) + 1
        print(f"  - {feature} (RF rank: {rf_rank}, PCA rank: {pca_rank})")
    
    # Create a combined importance score based on both methods
    # Normalize both importance metrics to 0-1 range
    feature_importance['RF_Importance_Normalized'] = feature_importance['Importance'] / feature_importance['Importance'].max()
    pca_feature_importance['PCA_Importance_Normalized'] = pca_feature_importance['PCA_Importance'] / pca_feature_importance['PCA_Importance'].max()
    
    # Merge the two dataframes
    combined_importance = pd.merge(
        feature_importance[['Feature', 'RF_Importance_Normalized']], 
        pca_feature_importance[['Feature', 'PCA_Importance_Normalized']], 
        on='Feature'
    )
    
    # Calculate combined score (average of the two normalized scores)
    combined_importance['Combined_Score'] = (combined_importance['RF_Importance_Normalized'] + 
                                            combined_importance['PCA_Importance_Normalized']) / 2
    
    # Sort by combined score
    combined_importance = combined_importance.sort_values('Combined_Score', ascending=False)
    
    # Plot combined feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Combined_Score', y='Feature', data=combined_importance.head(20))
    plt.title('Top 20 Features by Combined Importance (RF + PCA)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_combined.png")
    plt.show()

    # 8. Advanced preprocessing and ML pipeline setup
    print("\nStep 3: Building the ML pipeline")
    
    # 8.1 Define preprocessing strategies based on analysis
    
    # For numeric features: Use KNN imputation for features with moderate missing values
    # and standard scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", RobustScaler())  # RobustScaler is less sensitive to outliers
        ]
    )
    
    # For categorical features: Use most frequent imputation and one-hot encoding
    # Limit max_categories based on cardinality analysis
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, 
                                     max_categories=15)),  # Adjusted based on cardinality analysis
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder='drop'  # Drop columns not specified in transformers
    )
    
    # 9. Experiment with multiple modeling approaches
    
    # 9.1 Logistic Regression with SMOTE
    logreg_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(f_classif, k=20)),
            ("sampling", SMOTE(random_state=42)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000, 
                    class_weight="balanced",
                    solver="liblinear",
                    C=0.1
                ),
            ),
        ]
    )
    
    # 9.2 Random Forest with balanced class weights
    rf_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampling", SMOTE(random_state=42)),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                ),
            ),
        ]
    )
    
    # 9.3 Gradient Boosting
    gb_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                ),
            ),
        ]
    )
    
    # 10. Cross-validation and model evaluation
    print("\nStep 4: Model evaluation with cross-validation")
    
    # Define stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate models with cross-validation
    model_pipelines = {
        "Logistic Regression": logreg_pipeline,
        "Random Forest": rf_pipeline,
        "Gradient Boosting": gb_pipeline
    }
    
    results = {}
    
    for name, pipeline in model_pipelines.items():
        print(f"\nEvaluating {name}...")
        # Explicitly use balanced_accuracy_score for consistency
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        results[name] = {
            "scores": scores,
            "mean": np.mean(scores),
            "std": np.std(scores)
        }
        print(f"{name} CV Scores: {scores}")
        print(f"{name} Mean Balanced Accuracy: {results[name]['mean']:.4f} ± {results[name]['std']:.4f}")
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k]['mean'])
    best_model = model_pipelines[best_model_name]
    
    print(f"\nBest model: {best_model_name} with balanced accuracy: {results[best_model_name]['mean']:.4f}")
    
    # Add a model evaluation function to ensure consistent metrics
    def evaluate_model(model, X_data, y_true, dataset_name=""):
        """Evaluate model with balanced accuracy score and other metrics"""
        y_pred = model.predict(X_data)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        print(f"\nEvaluation on {dataset_name} dataset:")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix on {dataset_name} Dataset')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{dataset_name.lower()}.png")
        plt.show()
        
        return bal_acc
    
    # 11. Hyperparameter tuning for best model
    print("\nStep 5: Hyperparameter tuning for the best model")
    
    if best_model_name == "Logistic Regression":
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'feature_selection__k': [15, 20, 25]
        }
    elif best_model_name == "Random Forest":
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10]
        }
    else:  # Gradient Boosting
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [2, 3, 4]
        }
    
    grid_search = GridSearchCV(
        best_model,
        param_grid,
        cv=3,
        scoring='balanced_accuracy',  # Explicitly using balanced_accuracy
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    tuned_model = grid_search.best_estimator_
    
    print("Best parameters:", grid_search.best_params_)
    print(f"Tuned model balanced accuracy: {grid_search.best_score_:.4f}")
    
    # 12. Train final model on full dataset (train + test)
    print("\nStep 6: Training final model on combined training and test data")
    
    # Make sure 'y' column exists in test_df or create empty Series if it doesn't
    if 'y' not in test_df.columns:
        test_df['y'] = pd.Series([float('nan')] * len(test_df))
    
    # Combine train and test data
    X_full = pd.concat([X, test_df.drop('y', axis=1)])
    y_full = pd.concat([y, test_df['y']])
    
    # Drop NaN values from y_full if they exist
    valid_indices = ~y_full.isna()
    X_full_valid = X_full[valid_indices]
    y_full_valid = y_full[valid_indices]
    
    # Train the tuned model on valid data
    tuned_model.fit(X_full_valid, y_full_valid)
    
    # After fitting the model on the full dataset, evaluate it
    if len(X_full_valid) > 0 and len(y_full_valid) > 0:
        print("\nEvaluating the final model on the combined dataset:")
        final_bal_acc = evaluate_model(tuned_model, X_full_valid, y_full_valid, "Combined Training")
        print(f"Final model balanced accuracy on combined data: {final_bal_acc:.4f}")
    
    # 13. Make predictions on validation set
    print("\nStep 7: Making predictions on validation set")
    if has_validation:
        validation_preds = tuned_model.predict(validation_df)
        
        # Save predictions to file with the required format: index,y
        validation_results = pd.DataFrame({
            'y': validation_preds
        })
        
        output_path = "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/validation_predictions.csv"
        validation_results.to_csv(output_path, index=True)
        print(f"Predictions saved to {output_path} with format 'index,y'")
        
        # When making validation predictions, also evaluate if possible
        if 'y' in validation_df.columns and not validation_df['y'].isna().all():
            print("\nEvaluating the final model on the validation set:")
            val_bal_acc = evaluate_model(tuned_model, validation_df.drop('y', axis=1, errors='ignore'), 
                                       validation_df['y'], "Validation")
            print(f"Final model balanced accuracy on validation data: {val_bal_acc:.4f}")
    else:
        print("No validation data available for predictions")
    
    # 14. Feature importance of final model
    if hasattr(tuned_model[-1], 'feature_importances_'):
        try:
            # Get feature names after preprocessing
            feature_names = []
            
            # Extract feature names from preprocessor
            if hasattr(tuned_model.named_steps['preprocessor'], 'get_feature_names_out'):
                feature_names = tuned_model.named_steps['preprocessor'].get_feature_names_out()
            
            # If feature selection was used, filter feature names
            if 'feature_selection' in dict(tuned_model.named_steps):
                selected_indices = tuned_model.named_steps['feature_selection'].get_support(indices=True)
                feature_names = [feature_names[i] for i in selected_indices]
            
            # Get feature importances
            feature_importances = tuned_model[-1].feature_importances_
            
            # Create dataframe of feature importances
            if len(feature_names) == len(feature_importances):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=False)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
                plt.title('Top 20 Features by Importance (Final Model)')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/final_feature_importance.png")
                plt.show()
                
                # Also apply PCA on the transformed features
                X_transformed = tuned_model.named_steps['preprocessor'].transform(X)
                
                # Apply PCA to get another view of feature importance in final model
                try:
                    # Standardize if not already scaled
                    X_final_scaled = StandardScaler().fit_transform(X_transformed)
                    
                    # Apply PCA
                    final_pca = PCA(n_components=min(20, X_final_scaled.shape[1]))
                    final_pca.fit(X_final_scaled)
                    
                    # Print explained variance
                    explained_var = final_pca.explained_variance_ratio_
                    cumulative_var = np.cumsum(explained_var)
                    
                    # Plot the explained variance
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label='Individual')
                    plt.step(range(1, len(cumulative_var) + 1), cumulative_var, where='mid', label='Cumulative')
                    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Variance Threshold')
                    plt.xlabel('Principal Components')
                    plt.ylabel('Explained Variance Ratio')
                    plt.title('PCA Explained Variance')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/pca_explained_variance.png")
                    plt.show()
                    
                    # Determine how many components needed to explain 80% variance
                    n_components_80 = np.where(cumulative_var >= 0.8)[0][0] + 1
                    print(f"\nNumber of PCA components needed to explain 80% variance: {n_components_80}")
                    
                except Exception as pca_error:
                    print(f"Could not perform PCA on transformed features: {pca_error}")
        except Exception as e:
            print(f"Could not extract feature importances: {e}")
    
    print("\nAnalysis and model training completed successfully!")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback to a simpler model if the complex one fails
    try:
        print("\nTrying fallback model...")
        
        # If train_df wasn't loaded, try loading it first
        if train_df is None:
            train_df = pd.read_csv(
                "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/datasets/final/final_proj_data.csv"
            )
            
        # If test_df wasn't loaded, try loading it
        if test_df is None:
            try:
                test_df = pd.read_csv(
                    "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/datasets/final/final_proj_test.csv"
                )
            except:
                test_df = pd.DataFrame()  # Empty DataFrame if file doesn't exist
        
        # If validation_df wasn't loaded, try creating from test or use an empty DataFrame
        if validation_df is None:
            try:
                test_df, validation_df = train_test_split(test_df, test_size=0.3, random_state=42)
            except:
                if len(test_df) > 0:
                    # If test_df exists but split failed
                    validation_df = test_df.copy()
                else:
                    # If no test data available, use an empty DataFrame
                    validation_df = pd.DataFrame()
        
        # Prepare features and target if not already done
        if X is None or y is None:
            X = train_df.drop(columns=["y"], errors="ignore")
            y = train_df["y"]
            num_features = X.select_dtypes(include="number").columns.tolist()
            cat_features = X.select_dtypes(include="object").columns.tolist()
        
        # Simple preprocessing without feature engineering
        simple_preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ]
        )
        
        # Simple logistic regression model
        simple_pipeline = Pipeline([
            ('preprocessor', simple_preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        # Train on available data
        simple_pipeline.fit(X, y)
        
        # Include evaluation in the fallback path too
        fallback_bal_acc = evaluate_model(simple_pipeline, X, y, "Training (Fallback)")
        print(f"Fallback model balanced accuracy on training data: {fallback_bal_acc:.4f}")
        
        # Only make predictions if validation data is available
        if validation_df is not None and len(validation_df) > 0:
            fallback_preds = simple_pipeline.predict(validation_df)
            
            # Use the required format: index,y
            fallback_results = pd.DataFrame({
                'y': fallback_preds
            })
            
            fallback_path = "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/validation_predictions_fallback.csv"
            fallback_results.to_csv(fallback_path, index=True)
            print(f"Fallback predictions saved to {fallback_path} with format 'index,y'")
        else:
            print("No validation data available for predictions")
        
    except Exception as inner_e:
        print(f"Fallback also failed: {inner_e}")
        # Last resort - just predict the most common class
        try:
            most_common = train_df['y'].mode()[0]
            
            # Only create predictions if validation data exists
            if validation_df is not None and len(validation_df) > 0:
                # Use the required format: index,y
                last_resort_results = pd.DataFrame({
                    'y': [most_common] * len(validation_df)
                })
                
                last_resort_path = "/home/nickolasz/Projects/GoIT/MACHINE-LEARNING-NEO/validation_predictions_last_resort.csv"
                last_resort_results.to_csv(last_resort_path, index=True)
                print(f"Last resort predictions saved to {last_resort_path} with format 'index,y'")
            else:
                print("No validation data available for predictions")
        except:
            print("Could not create last resort predictions")
