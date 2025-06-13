import os
import argparse
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

def convert_joblib_to_onnx(models_dir: str, output_dir: str, model_file: str):
    """Converts all Joblib models in subdirectories to ONNX format."""
    os.makedirs(output_dir, exist_ok=True)
    converted_count = 0
    
    print(f"Starting conversion from: {models_dir}")
    for model_name in os.listdir(models_dir):
        actual_filename = model_file.replace("{model_name}", model_name)
        model_path = os.path.join(models_dir, model_name, actual_filename)
        
        if not os.path.isfile(model_path):
            print(f"  ⚠️ File not found: {model_path}, skipping")
            continue
        
        print(f"  Converting model: {model_name}")
        
        try:
            model_data = joblib.load(model_path)
            
            if 'pipeline' not in model_data:
                raise ValueError("Joblib file doesn't contain 'pipeline' key")
                
            pipeline = model_data['pipeline']
            
            initial_type = [("input", StringTensorType([None, 1]))]
            
            model_type = model_data.get('model_type', 'UnknownModel')
            onnx_model = convert_sklearn(
                pipeline,
                initial_types=initial_type,
                options={'zipmap': False},
                name=f"{model_name}_{model_type}"
            )
            
            output_path = os.path.join(output_dir, f"{model_name}.onnx")
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"  ✅ Saved ONNX model to: {output_path}")
            converted_count += 1
            
        except Exception as e:
            print(f"  ❌ Conversion failed for {model_name}: {str(e)}")
    
    print(f"\nConversion complete! {converted_count} models converted successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert scikit-learn joblib models to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--models_dir", 
                        default="models", 
                        help="Directory containing model subdirectories")
    parser.add_argument("--output_dir", 
                        default="onnx_models", 
                        help="Directory to save converted ONNX models")
    parser.add_argument("--model_file", 
                        default="best_model_{model_name}.joblib", 
                        help="Joblib filename template with {model_name} placeholder")
    
    args = parser.parse_args()
    
    convert_joblib_to_onnx(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        model_file=args.model_file
    )