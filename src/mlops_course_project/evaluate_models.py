import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import logging
import time
from pathlib import Path
import os
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_models(model_test_pairs, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    
    for model_path, test_df in model_test_pairs:
        try:
            start_time = time.time()
            
            if not {'text', 'topic'}.issubset(test_df.columns):
                raise ValueError("Тестовые данные должны содержать колонки 'text'и 'topic'")
            
            model_data = joblib.load(model_path)
            pipeline = model_data['pipeline']
            model_type = model_data.get('model_type', 'unknown')
            dataset_name = model_data.get('dataset_name', 'unknown')
            
            logging.info(f"\n{'='*40}")
            logging.info(f"Валидация {model_type} модели (датасет {dataset_name})")
            logging.info(f"Размер тестовой выборки: {len(test_df):,}")
            logging.info(f"Путь к модели: {model_path}")
            logging.info(f"{'='*40}")
            
            X_test = test_df['text']
            y_true = test_df['topic']
            
            pred_start = time.time()
            y_pred = pipeline.predict(X_test)
            pred_time = time.time() - pred_start
            
            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            
            result = {
                'model_type': model_type,
                'dataset_name': dataset_name,
                'test_samples': len(test_df),
                'prediction_time': pred_time,
                'accuracy': report['accuracy'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'confusion_matrix': cm,
                'model_path': model_path,
                'full_report': report
            }
            results.append(result)
            
            report_path = Path(output_dir) / f"dataset_{dataset_name}_{model_type}_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Тип модели: {model_type}\n")
                f.write(f"Номер датасета: {dataset_name}\n")
                f.write(f"Количество объектов в тестовой выборке: {len(test_df):,}\n")
                f.write(f"Затраченное время: {pred_time:.2f}s\n")
                f.write("\nClassification Report:\n")
                f.write(classification_report(y_true, y_pred))
                f.write("\nConfusion Matrix:\n")
                f.write(str(cm))
            
            logging.info(f"Валидация заняла {time.time() - start_time:.2f}s")
            logging.info(f"Результаты сохранены в {report_path}")
            
        except Exception as e:
            logging.error(f"Ошибка валидации {model_path}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

    model_test_pairs = []
    models_path = Path(cfg.evaluate.models_dir)

    for model_dir in models_path.iterdir():
        if model_dir.is_file():
            model_file_path = model_dir.name
            dataset_name = model_file_path.split("_")[2].split(".")[0]
            test_data = pd.read_csv(os.path.join(cfg.evaluate.processed_datasets_dir, dataset_name, "test_df.csv")).dropna()
            model_test_pairs.append((os.path.join(models_path, model_file_path), test_data))

    output_dir = Path(cfg.evaluate.validation_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = evaluate_models(model_test_pairs, output_dir)
    results_df.to_csv(os.path.join(cfg.evaluate.validation_results_dir, cfg.evaluate.result_file_name), index=False)

if __name__ == "__main__":
    main()