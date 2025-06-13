import json
import numpy as np
import triton_python_backend_utils as pb_utils
import nltk
from cleantext import clean
import os
from pathlib import Path

class TritonPythonModel:
    """Triton Python Model implementing full 'soft' processing pipeline"""

    def initialize(self, args):
        """Initialize the model with soft processing parameters"""
        self.logger = pb_utils.Logger
        self.model_dir = Path(args['model_repository'])
        self.version_dir = self.model_dir / args['model_version']
        
        # Load class labels
        # class_labels_path = self.version_dir / 'class_labels.txt'
        # with open(class_labels_path, 'r') as f:
        #     self.class_labels = [line.strip() for line in f.readlines()]
        
        # Load processing configuration
        config_path = self.version_dir / 'processing_config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize Russian stopwords
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        self.russian_stopwords = set(nltk.corpus.stopwords.words('russian'))
        
        self.logger.log("Soft processing model initialized")
        self.logger.log(f"Loaded configuration: {json.dumps(self.config, indent=2)}")

    def execute(self, requests):
        """Execute full processing pipeline for batch of requests"""
        responses = []
        
        for request in requests:
            # Get input text
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_texts = [t[0].decode("UTF-8") for t in input_tensor.as_numpy()]
            
            # Preprocess texts
            preprocessed_texts = []
            for text in input_texts:
                # Step 1: Clean text
                cleaned_text = clean(
                    text,
                    fix_unicode=self.config['fix_unicode'],
                    to_ascii=self.config['to_ascii'],
                    lower=self.config['lower'],
                    no_line_breaks=self.config['no_line_breaks'],
                    no_urls=self.config['no_urls'],
                    no_emails=self.config['no_emails'],
                    no_phone_numbers=self.config['no_phone_numbers'],
                    no_numbers=self.config['no_numbers'],
                    no_digits=self.config['no_digits'],
                    no_currency_symbols=self.config['no_currency_symbols'],
                    no_punct=self.config['no_punct'],
                    lang=self.config['lang']
                )
                
                # Step 2: Tokenize and filter
                tokens = nltk.word_tokenize(cleaned_text, language='russian')
                tokens = [t for t in tokens if len(t) > 1]  # Remove single-character tokens
                
                preprocessed_texts.append(" ".join(tokens))
            
            # Prepare input for ONNX model
            onnx_input = np.array([[text] for text in preprocessed_texts], dtype=object)
            
            # Create inference request to ONNX model
            inference_request = pb_utils.InferenceRequest(
                model_name='inference_soft',
                requested_output_names=['label'],
                inputs=[pb_utils.Tensor("input", onnx_input)]
            )
            inference_response = inference_request.exec()
            
            # Handle inference errors
            if inference_response.has_error():
                error_msg = f"ONNX inference failed: {inference_response.error().message()}"
                self.logger.log(error_msg, pb_utils.Logger.ERROR)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(error_msg)
                    )
                )
                continue
            
            # Get output tensor from ONNX model
            output_tensor = pb_utils.get_output_tensor_by_name(inference_response, "label")
            topic = output_tensor.as_numpy()
            
            # Map class indices to labels
            # output_labels = [[self.class_labels[idx].encode("UTF-8")] for idx in class_indices]
            print(topic)
            # output_labels = [topic for topic in class_indices]
            
            # Create response tensor
            output_tensor = pb_utils.Tensor(
                "OUTPUT", 
                np.array(topic, dtype=object)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output_tensor])
            )
        
        return responses

    def finalize(self):
        """Cleanup resources"""
        self.logger.log("Cleaning up soft processing model")