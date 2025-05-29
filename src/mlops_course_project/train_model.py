import optuna
from optuna.samplers import TPESampler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg.training
        self.study = None

    def _create_vectorizer(self, trial, vectorizer_type):
        vec_cfg = self.cfg.vectorizers[vectorizer_type]
        params = {}
        
        for param, spec in vec_cfg.items():
            if spec.type == "categorical":
                params[param] = trial.suggest_categorical(f"{vectorizer_type}__{param}", spec.options)
            elif spec.type == "float":
                params[param] = trial.suggest_float(f"{vectorizer_type}__{param}", 
                                                  spec.low, spec.high, log=spec.get("log", False))
            elif spec.type == "int":
                params[param] = trial.suggest_int(f"{vectorizer_type}__{param}", 
                                                spec.low, spec.high)
        
        return TfidfVectorizer(**params) if vectorizer_type == "tfidf" else CountVectorizer(**params)

    def _create_classifier(self, trial, model_type):
        model_cfg = self.cfg.models[model_type]
        params = {}
        
        for param, spec in model_cfg.items():
            if param == "type":
                continue
            if spec.type == "categorical":
                params[param] = trial.suggest_categorical(f"{model_type}__{param}", spec.options)
            elif spec.type == "float":
                params[param] = trial.suggest_float(f"{model_type}__{param}", 
                                                  spec.low, spec.high, log=spec.get("log", False))
            elif spec.type == "int":
                params[param] = trial.suggest_int(f"{model_type}__{param}", 
                                                spec.low, spec.high)
        
        if model_type == "logreg":
            return LogisticRegression(solver='liblinear', class_weight='balanced', **params)
        elif model_type == "svm":
            return SVC(class_weight='balanced', probability=True, **params)
        else:
            return RandomForestClassifier(class_weight='balanced', **params)

    def _objective(self, trial, X, y):
        vectorizer_type = trial.suggest_categorical("vectorizer_type", self.cfg.vectorizers.vectorizer_types)
        vectorizer = self._create_vectorizer(trial, vectorizer_type)
        
        model_type = trial.suggest_categorical("model_type", self.cfg.models.model_types)
        classifier = self._create_classifier(trial, model_type)
        
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

        cv = StratifiedKFold(n_splits=self.cfg.optuna_pars.n_folds, 
                           shuffle=True, 
                           random_state=self.cfg.optuna_pars.random_seed)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average='weighted'))
            
        return np.mean(scores)

    def optimize(self, X, y):
        sampler = TPESampler(seed=self.cfg.optuna_pars.random_seed)
        self.study = optuna.create_study(
            direction="maximize", 
            sampler=sampler,
            study_name=self.cfg.optuna_pars.get("study_name", "optimization_study")
        )
        self.study.optimize(
            lambda trial: self._objective(trial, X, y), 
            n_trials=self.cfg.optuna_pars.n_trials
        )
        
        logging.info(f"Best trial: {self.study.best_trial.value}")
        logging.info(f"Best params: {self.study.best_trial.params}")
        
        return self._create_best_model()

    def _create_best_model(self):
        best_params = self.study.best_params
        vectorizer_type = best_params["vectorizer_type"]
        model_type = best_params["model_type"]
        
        vectorizer = self._create_vectorizer(optuna.trial.FixedTrial(best_params), vectorizer_type)
        classifier = self._create_classifier(optuna.trial.FixedTrial(best_params), model_type)
        
        return Pipeline([('vectorizer', vectorizer), ('classifier', classifier)]), model_type

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model_dir = Path(to_absolute_path(cfg.paths.models_dir)) / cfg.processing_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
    # OmegaConf.save(cfg, experiment_dir / "config.yaml")
    
    # datasets = []
    # dataset_path = 
    dataset_dir = Path(to_absolute_path(cfg.paths.processed_data)) / cfg.processing_name
    # for dataset_dir in base_path.iterdir():
    if dataset_dir.is_dir():
        try:
            dataset_name = cfg.processing_name
            train_df = pd.read_csv(dataset_dir / 'train_df.csv').sample(cfg.dataset.sample_size).dropna()
            val_df = pd.read_csv(dataset_dir / 'val_df.csv').sample(cfg.dataset.sample_size).dropna()
            test_df = pd.read_csv(dataset_dir / 'test_df.csv').sample(cfg.dataset.sample_size).dropna()
            
            # datasets.append((dataset_name, train_df, val_df, test_df))
            logging.info(f"Loaded dataset: {dataset_name}")
            
        except Exception as e:
            logging.warning(f"Skipped {dataset_dir.name}: {str(e)}")

    # for dataset_name, train_df, val_df, test_df in datasets:
    logging.info(f"\n{'='*40}")
    logging.info(f"Processing dataset: {dataset_name}")
    
    X_train = pd.concat([train_df['text'], val_df['text']])
    y_train = pd.concat([train_df['topic'], val_df['topic']])
    
    optimizer = ModelOptimizer(cfg)
    best_pipeline, model_type = optimizer.optimize(X_train, y_train)
    best_pipeline.fit(X_train, y_train)
    
    model_data = {
        'pipeline': best_pipeline,
        'model_type': model_type,
        'dataset_name': dataset_name,
        'training_date': datetime.now().isoformat(),
        'config': OmegaConf.to_container(cfg)
    }

    model_path = model_dir / f"best_model_{dataset_name}.joblib"
    joblib.dump(model_data, model_path)
    logging.info(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()