import time
import numpy as np
import tritonclient.http as httpclient
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm

MODELS = ["soft", "medium", "rigid"]
RESULTS_FILE = "performance_report.md"

def test_model_performance(client, model_name, test_data):
    texts = test_data["text"].tolist()
    true_labels = test_data["topic"].tolist()
    
    latencies = []
    predictions = []
    
    for text in tqdm(texts):
        inputs = [httpclient.InferInput("input", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[text]], dtype=object))
        response = client.infer(model_name, inputs)
        output = response.as_numpy("label")
        predictions.append(output[0][0].decode())
    
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "f1": f1_score(true_labels, predictions, average="weighted")
    }

if __name__ == "__main__":
    client = httpclient.InferenceServerClient(url="localhost:8000")
    report = "# Model Performance Report\n\n"
    report += "| Model | Accuracy | F1-Score |\n"
    report += "|-------|----------|----------|\n"
    
    for model in MODELS:
        print(f"Testing {model} model...")
        test_data_path = f"./data/processed/{model}/test_df.csv"
        test_data = pd.read_csv(test_data_path)
        metrics = test_model_performance(client, model, test_data)
        
        report += (
            f"| {model} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} |\n"
        )
    
    with open(RESULTS_FILE, "w") as f:
        f.write(report)
    
    print(f"Report generated: {RESULTS_FILE}")