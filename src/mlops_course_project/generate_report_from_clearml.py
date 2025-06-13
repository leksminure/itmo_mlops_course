from clearml import Task
import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import os

def read_artifact(path):
    ext = os.path.splitext(path)[1]
    if ext == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'r') as f:
            return f.read().strip()

def generate_report(project_name, metric_name, output_dir="reports", tags_to_filter=["inference"]):
    tasks = Task.query_tasks(
        project_name=project_name,
        tags=tags_to_filter,
        task_filter={'status': ['completed']}
    )
    
    results = []
    for task in tasks:
        try:
            t = Task.get_task(task_id=task)

            model_type_path = t.artifacts['model_type'].get_local_copy()
            dataset_name_path = t.artifacts['dataset_name'].get_local_copy()
            metric_path = t.artifacts[metric_name].get_local_copy()

            model_type = read_artifact(model_type_path)
            dataset_name = read_artifact(dataset_name_path)
            metric_value = read_artifact(metric_path)

            if metric_value is not None:
                results.append({
                    'model_type': model_type,
                    'dataset_name': dataset_name,
                    metric_name: metric_value,
                    'task_id': t.task_id,
                    'status': t.status
                })
        except Exception as e:
            print(f"Error processing task {task}: {str(e)}")
    
    if not results:
        print(f"No evaluation tasks found")
        return
        
    df = pd.DataFrame(results)
    best_idx = df[metric_name].idxmax()
    best_dataset = df.loc[best_idx, 'dataset_name']
    best_model = df.loc[best_idx, 'model_type']
    best_metric = df.loc[best_idx, metric_name]
    
    report_content = [
        f"ClearML Evaluation Report ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})",
        f"Analyzed {len(df)} tasks",
        "",
        f"Best {metric_name} scores by dataset:"
    ]
    
    report_content.extend([
        "",
        f"Overall best {metric_name}: {float(best_metric):.4f}",
        f"Best preprocessing: {best_dataset}",
        f"Best model type: {best_model}",
        f"Task ID: {df.loc[best_idx, 'task_id']}"
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"clearml_report_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("\n".join(report_content))
    
    print(f"Report generated: {filepath}")
    return filepath

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    output_dir = Path(cfg.generate_report.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = cfg.generate_report.metric
    tags_to_filter = cfg.generate_report.tags
    project_name = cfg.generate_report.project_name
    
    generate_report(project_name, metric, output_dir, tags_to_filter)


if __name__ == "__main__":
    main()
    