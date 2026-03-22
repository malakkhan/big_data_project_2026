"""
Algorithmic Modeling and Evaluation Engine.

Utilizes XGBoost alongside Optuna for Bayesian Optimization.
Strictly implements Stratified K-Fold Cross Validation and Early Stopping
to prevent overfitting. Automatically tracks and saves ROC curves, confusion
matrices, and rigorous statistical metrics for every hyperparameter variation.

Classes:
    XGBoostModeler: Primary engine mapping features into the gradient boosted model.
"""

import sys
import logging
import joblib
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score, 
    precision_score, recall_score, ConfusionMatrixDisplay
)

from src import config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class XGBoostModeler:
    """
    Orchestrates the XGBoost training lifecycle including hyperparameter tuning.

    Attributes:
        experiment_prefix (str): Label defining the macro environment state.
        best_params (dict): The resolved optima for the XGB estimators.
        final_model (xgb.XGBClassifier): The finalized trained predictor.
    """

    def __init__(self, experiment_prefix="default", massive=False):
        """
        Initializes the modeler.

        Args:
            experiment_prefix (str): Identifier used to save experiment metadata 
                                     and precision visuals into `experiment_results/`.
        """
        self.experiment_prefix = experiment_prefix
        self.massive = massive
        self.best_params = None
        self.final_model = None
        self._best_oof_auc = 0.0
        self._best_oof_data = None  # (indices, y_true, y_probs) from best CV trial

    def prep_data(self):
        """
        Loads and prepares the fully imputed feature dataframe.

        Reads from the localized parity state (Parquet file), drops unstructured 
        identifiers, and isolates the target matrix vectors.

        Returns:
            tuple: (X, y)
                X (pd.DataFrame): The feature matrix.
                y (pd.Series): The target binary labels.
        """
        input_parquet = config.PARQUET_DIR / "imputed_features.parquet"
        df = pd.read_parquet(input_parquet)
        
        drop_cols = ["tconst", "synthetic_index", "primaryTitle", "originalTitle", "C1", "tmdb_success"]
        feature_cols = [c for c in df.columns if c not in drop_cols and c != "label"]
        
        X = df[feature_cols].copy()
        
        # XGBoost explicitly requires unstructured schema tensors to be categorically mapped
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna("Unknown").astype('category')
            
        y = df["label"].astype(int)
        return X, y

    def plot_and_save_artifacts(self, trial_number, y_true, y_probs, auc_score, params, feature_importances=None):
        """
        Generates and saves visual matrices and statistical outputs.

        Creates an explicit ROC Curve, a normalized Confusion Matrix, and JSON serialization.

        Args:
            trial_number (int): Optuna iteration index.
            y_true (np.array): Chronological truth labels.
            y_probs (np.array): Raw continuous probabilistic outputs.
            auc_score (float): Aggregate ROC-AUC score.
            params (dict): Param configuration bindings.
            feature_importances (dict): Optional dictionary mapping feature titles to Information Gain.
        """
        experiment_dir = config.OUTPUT_DIR / "experiment_results"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        trial_id = f"{self.experiment_prefix}_trial{trial_number}"
        
        y_pred = (y_probs >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        sens = recall_score(y_true, y_pred, zero_division=0) # sensitivity
        
        cm_matrix = confusion_matrix(y_true, y_pred)
        
        if cm_matrix.shape == (2, 2):
            tn, fp, fn, tp = cm_matrix.ravel()
        else:
            # Handle edge cases where one class might be missing in validation splits
            tn, fp, fn, tp = 0, 0, 0, 0
            
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics = {
            "roc_auc": float(auc_score),
            "accuracy": float(acc),
            "precision": float(prec),
            "sensitivity": float(sens),
            "specificity": float(spec)
        }
        
        output_payload = {
            "metrics": metrics,
            "hyperparameters": params,
            "confusion_matrix": {
                "True_Positives": int(tp),
                "False_Positives": int(fp),
                "True_Negatives": int(tn),
                "False_Negatives": int(fn)
            }
        }
        
        if feature_importances is not None:
            output_payload["feature_importances_gain"] = feature_importances
            
            # Plot Feature Importance Bar Chart
            plt.figure(figsize=(10, 8))
            fi_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
            # Sort bottom-up for horizontal bar charting, keeping the top 15
            fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(15)
            
            plt.barh(fi_df['Feature'], fi_df['Importance'], color='teal')
            plt.xlabel('Information Gain')
            plt.title(f'XGBoost Feature Importances (Top 15) - {trial_id}')
            
            plot_path_fi = experiment_dir / f"{trial_id}_feature_importance.png"
            plt.savefig(plot_path_fi, bbox_inches='tight')
            plt.close()
            
        # Save JSON Stats & Parameters
        with open(experiment_dir / f"{trial_id}.json", "w") as f:
            json.dump(output_payload, f, indent=4)
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {trial_id}')
        plt.legend(loc="lower right")
        
        plot_path_roc = experiment_dir / f"{trial_id}_roc.png"
        plt.savefig(plot_path_roc, bbox_inches='tight')
        plt.close()

        # Plot Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
        plt.title(f'Confusion Matrix Grid - {trial_id}')
        
        plot_path_cm = experiment_dir / f"{trial_id}_cm.png"
        plt.savefig(plot_path_cm, bbox_inches='tight')
        plt.close()

    def objective(self, trial, X, y):
        """
        Optuna objective function for K-Fold CV Bayesian optimization.

        Args:
            trial (optuna.trial.Trial): The active tuning trial.
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.

        Returns:
            float: The aggregate out-of-fold ROC-AUC score.
        """
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', config.XGB_PARAM_BOUNDS['n_estimators'][0], config.XGB_PARAM_BOUNDS['n_estimators'][1]),
            'learning_rate': trial.suggest_float('learning_rate', config.XGB_PARAM_BOUNDS['learning_rate'][0], config.XGB_PARAM_BOUNDS['learning_rate'][1], log=True),
            'max_depth': trial.suggest_int('max_depth', config.XGB_PARAM_BOUNDS['max_depth'][0], config.XGB_PARAM_BOUNDS['max_depth'][1]),
            'reg_alpha': trial.suggest_float('reg_alpha', config.XGB_PARAM_BOUNDS['reg_alpha'][0], config.XGB_PARAM_BOUNDS['reg_alpha'][1]), 
            'reg_lambda': trial.suggest_float('reg_lambda', config.XGB_PARAM_BOUNDS['reg_lambda'][0], config.XGB_PARAM_BOUNDS['reg_lambda'][1]), 
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
        
        oof_indices = []
        oof_y_true = []
        oof_y_probs = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # Protect testing purity! Sub-split the training set strictly for early stopping evaluation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=42
            )
            
            model = xgb.XGBClassifier(**param, early_stopping_rounds=20, enable_categorical=True)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict_proba(X_test)[:, 1]
            oof_indices.extend(test_idx)
            oof_y_true.extend(y_test)
            oof_y_probs.extend(preds)
            
        final_auc = roc_auc_score(oof_y_true, oof_y_probs)
        
        # Track the best trial's out-of-fold predictions for honest misclassification reporting
        if final_auc > self._best_oof_auc:
            self._best_oof_auc = final_auc
            self._best_oof_data = (
                np.array(oof_indices),
                np.array(oof_y_true),
                np.array(oof_y_probs),
            )
        
        # Extract native Information Gain measurements from the culminated K-Fold predictor
        f_list = list(X.columns)
        importances = model.feature_importances_
        fi_dict = {f_list[i]: float(importances[i]) for i in range(len(f_list))}
        
        self.plot_and_save_artifacts(
            trial.number, 
            np.array(oof_y_true), 
            np.array(oof_y_probs), 
            final_auc, 
            param, 
            feature_importances=fi_dict
        )
            
        return final_auc

    def _run_massive(self):
        """
        PySpark Native distributed optimization loop for Massive Datasets.
        """
        logger.info(f"[{self.experiment_prefix}] Initializing MASSIVE Distributed PySpark XGBoost Optimization...")
        
        from pyspark.sql import SparkSession
        import pyspark.sql.types as T
        from pyspark.ml.feature import VectorAssembler, StringIndexer
        from pyspark.ml import Pipeline
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
        from xgboost.spark import SparkXGBClassifier
        
        spark = SparkSession.builder.appName("MassiveOptunaXGB").getOrCreate()
        df = spark.read.parquet(str(config.PARQUET_DIR / "imputed_features.parquet"))
        
        drop_cols = ["tconst", "synthetic_index", "primaryTitle", "originalTitle", "C1", "tmdb_success"]
        feature_cols = [c for c in df.columns if c not in drop_cols and c != "label"]
        
        categorical_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType) and f.name in feature_cols]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]
        
        # Native MLlib encoding pipeline for XGBoost VectorAssembler
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
        indexed_cat_cols = [f"{c}_idx" for c in categorical_cols]
        
        # Cast numeric cols to double to avoid datatype mismatch in VectorAssembler
        from pyspark.sql.functions import col
        for c in numeric_cols:
            df = df.withColumn(c, col(c).cast(T.DoubleType()))
        
        assembler = VectorAssembler(inputCols=numeric_cols + indexed_cat_cols, outputCol="features", handleInvalid="keep")
        pipeline = Pipeline(stages=indexers + [assembler])
        
        prep_model = pipeline.fit(df)
        prepared_df = prep_model.transform(df)
        
        # Cast label to double just in case
        prepared_df = prepared_df.withColumn("label", col("label").cast(T.DoubleType()))
        
        def massive_objective(trial):
            param = {
                'features_col': 'features',
                'label_col': 'label',
                'n_estimators': trial.suggest_int('n_estimators', config.XGB_PARAM_BOUNDS['n_estimators'][0], config.XGB_PARAM_BOUNDS['n_estimators'][1]),
                'learning_rate': trial.suggest_float('learning_rate', config.XGB_PARAM_BOUNDS['learning_rate'][0], config.XGB_PARAM_BOUNDS['learning_rate'][1], log=True),
                'max_depth': trial.suggest_int('max_depth', config.XGB_PARAM_BOUNDS['max_depth'][0], config.XGB_PARAM_BOUNDS['max_depth'][1]),
                'reg_alpha': trial.suggest_float('reg_alpha', config.XGB_PARAM_BOUNDS['reg_alpha'][0], config.XGB_PARAM_BOUNDS['reg_alpha'][1]), 
                'reg_lambda': trial.suggest_float('reg_lambda', config.XGB_PARAM_BOUNDS['reg_lambda'][0], config.XGB_PARAM_BOUNDS['reg_lambda'][1])
            }
            
            xgb_est = SparkXGBClassifier(**param)
            evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
            
            cv = CrossValidator(estimator=xgb_est,
                                estimatorParamMaps=ParamGridBuilder().build(),
                                evaluator=evaluator,
                                numFolds=3,
                                seed=42)
            
            cv_model = cv.fit(prepared_df)
            return cv_model.avgMetrics[0]
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        # Run 5 iterations on massive to prove architecture (scale to 25 later)
        study.optimize(massive_objective, n_trials=5)
        
        self.best_params = study.best_params
        logger.info(f"Massive Optimization finished. Best PySpark ROC-AUC: {study.best_value:.4f}")
        logger.info(f"Best Distributed Hyperparameters: {self.best_params}")
        
        # Final Full-Pass Distributed Retrain
        final_params = {'features_col': 'features', 'label_col': 'label', **self.best_params}
        final_xgb = SparkXGBClassifier(**final_params)
        final_model = final_xgb.fit(prepared_df)
        
        model_path = str(config.OUTPUT_DIR / "models" / f"{self.experiment_prefix}_sparkxgb_best")
        final_model.write().overwrite().save(model_path)
        logger.info(f"Final Massive Distributed Model mapped to PySpark Directory: {model_path}")
        
        return study.best_value

    def run(self):
        """
        Executes the optimization framework and retrains the champion model.

        Fires Optuna trials optimizing the ROC-AUC. Captures the best hyperparameters
        and reconstructs an unrestrained final predictor, saving it to disk for the
        MLOps module.

        Returns:
            None
        """
        if self.massive:
            return self._run_massive()
            
        logger.info(f"[{self.experiment_prefix}] Initializing XGBoost Optimization...")
        
        try:
            X, y = self.prep_data()
        except FileNotFoundError:
            logger.error("Imputed features parquet not found. Ensure prior pipeline stages ran.")
            return
            
        dataset_size = len(X)
        logger.info(f"Training dataset size: {dataset_size} rows, {X.shape[1]} features.")
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=25) 
        
        self.best_params = study.best_params
        logger.info(f"Optimization finished. Best ROC-AUC: {study.best_value:.4f}")
        logger.info(f"Best Hyperparameters: {self.best_params}")
        
        final_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'random_state': 42,
            **self.best_params
        }
        self.final_model = xgb.XGBClassifier(**final_params, enable_categorical=True)
        self.final_model.fit(X, y) 
        
        model_path = config.OUTPUT_DIR / "models" / f"{self.experiment_prefix}_xgboost_best.joblib"
        joblib.dump(self.final_model, model_path)
        
        feature_cols_path = config.OUTPUT_DIR / "models" / f"{self.experiment_prefix}_feature_schema.json"
        pd.Series(X.columns).to_json(feature_cols_path, orient="records")
        
        categories_dict = {col: list(X[col].cat.categories) for col in X.select_dtypes(include=['category']).columns}
        with open(config.OUTPUT_DIR / "models" / f"{self.experiment_prefix}_categorical_maps.json", "w") as f:
            json.dump(categories_dict, f)
        
        # Persist out-of-fold predictions from the best CV trial for honest
        # misclassification reporting (not re-predicting on training data).
        if self._best_oof_data is not None:
            oof_indices, oof_true, oof_probs = self._best_oof_data
            oof_df = pd.DataFrame({
                "row_index": oof_indices,
                "true_label": oof_true,
                "predicted_prob": oof_probs,
            })
            oof_path = config.OUTPUT_DIR / "models" / f"{self.experiment_prefix}_oof_predictions.parquet"
            oof_df.to_parquet(oof_path, index=False)
            logger.info(f"Out-of-fold predictions saved to {oof_path}")
        
        logger.info(f"Final Model mapped to {model_path}")
        return study.best_value

if __name__ == "__main__":
    modeler = XGBoostModeler()
    modeler.run()
