import optuna
from optuna.samplers import TPESampler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RANDOM_STATE = 42

class ModelOptimizer:
    def __init__(self, n_trials=50, n_folds=5):
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.study = None

    def _objective(self, trial, X, y):
        vectorizer_type = trial.suggest_categorical("vectorizer_type", ["tfidf", "count"])
        
        if vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=trial.suggest_categorical("tfidf__ngram_range", [(1, 1), (1, 2)]),
                max_df=trial.suggest_float("tfidf__max_df", 0.75, 1.0),
                min_df=trial.suggest_int("tfidf__min_df", 1, 5),
                use_idf=trial.suggest_categorical("tfidf__use_idf", [True, False])
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=trial.suggest_categorical("count__ngram_range", [(1, 1), (1, 2)]),
                max_df=trial.suggest_float("count__max_df", 0.75, 1.0),
                min_df=trial.suggest_int("count__min_df", 1, 5)
            )

        model_type = trial.suggest_categorical("model_type", ["logreg", "svm", "random_forest"])
        
        if model_type == "logreg":
            classifier = LogisticRegression(
                solver='liblinear',
                C=trial.suggest_float("logreg__C", 1e-2, 1e2, log=True),
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
        elif model_type == "svm":
            classifier = SVC(
                C=trial.suggest_float("svm__C", 1e-2, 1e2, log=True),
                kernel=trial.suggest_categorical("svm__kernel", ["linear", "rbf"]),
                class_weight='balanced',
                probability=True,
                random_state=RANDOM_STATE
            )
        else:
            classifier = RandomForestClassifier(
                n_estimators=trial.suggest_int("rf__n_estimators", 100, 500),
                max_depth=trial.suggest_int("rf__max_depth", 5, 50),
                class_weight='balanced',
                random_state=RANDOM_STATE
            )

        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average='weighted'))
            
        return np.mean(scores)

    def optimize(self, X, y):
        sampler = TPESampler(seed=RANDOM_STATE)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self.study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)
        
        logging.info(f"Best trial:")
        trial = self.study.best_trial
        logging.info(f"Value: {trial.value}")
        logging.info(f"Params: {trial.params}")
        
        return self._create_best_model(trial.params)

    def _create_best_model(self, best_params):
        vectorizer_type = best_params["vectorizer_type"]
        vectorizer_params = {k.split("__", 1)[1]: v 
                           for k, v in best_params.items() 
                           if k.startswith(vectorizer_type)}
        
        vectorizer = (TfidfVectorizer(**vectorizer_params) if vectorizer_type == "tfidf" 
                     else CountVectorizer(**vectorizer_params))
        
        model_type = best_params["model_type"]
        classifier_params = {k.split("__", 1)[1]: v 
                            for k, v in best_params.items() 
                            if k.startswith(model_type.split("_")[0])}
        
        if model_type == "logreg":
            classifier = LogisticRegression(
                solver='liblinear',
                class_weight='balanced',
                random_state=RANDOM_STATE,
                **classifier_params
            )
        elif model_type == "svm":
            classifier = SVC(
                class_weight='balanced',
                probability=True,
                random_state=RANDOM_STATE,
                **classifier_params
            )
        else:
            classifier = RandomForestClassifier(
                class_weight='balanced',
                random_state=RANDOM_STATE,
                **classifier_params
            )
        
        return (
            Pipeline([('vectorizer', vectorizer), ('classifier', classifier)]),
            model_type
        )

def main(datasets, output_dir):
    for dataset_name, train_df, val_df, test_df in datasets:
        logging.info(f"\n{'='*40}")
        logging.info(f"Обрабатывается датасет {dataset_name}")
        logging.info(f"{'='*40}")
        
        X_train = pd.concat([train_df['text'], val_df['text']])
        y_train = pd.concat([train_df['topic'], val_df['topic']])
        
        optimizer = ModelOptimizer(n_trials=50, n_folds=5)
        best_pipeline, model_type = optimizer.optimize(X_train, y_train)
        best_pipeline.fit(X_train, y_train)
        
        model_data = {
            'pipeline': best_pipeline,
            'model_type': model_type,
            'dataset_name': dataset_name,
            'training_date': datetime.now().isoformat(),
            'features': 'text',
            'target': 'topic'
        }
        
        model_path = output_dir / f"best_model_{dataset_name}.joblib"
        joblib.dump(model_data, model_path)
        logging.info(f"Модель {model_type} сохранена для датасета {dataset_name} по пути {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', default='models')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    base_path = Path(args.input_path)

    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            try:
                train_df = pd.read_csv(dataset_dir / 'train_df.csv').sample(1000).dropna()
                val_df = pd.read_csv(dataset_dir / 'val_df.csv').sample(1000).dropna()
                test_df = pd.read_csv(dataset_dir / 'test_df.csv').sample(1000).dropna()
                
                datasets.append((dataset_name, train_df, val_df, test_df))
                logging.info(f"Загружен датасет: {dataset_name}")
                logging.info(f"Размер тренировочной выборки: {train_df.shape}")
                logging.info(f"Размер валидационной выборки: {val_df.shape}")
                logging.info(f"Размер тестовой выборки: {test_df.shape}")
                logging.info(f"{'='*40}")

                
            except Exception as e:
                logging.warning(f"Датасет {dataset_name} пропущен из-за: {str(e)}")

    main(datasets, output_dir)