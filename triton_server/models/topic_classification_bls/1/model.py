import json
import numpy as np
import triton_python_backend_utils as pb_utils
import nltk
from cleantext import clean
import os
from pathlib import Path
import pymorphy3

class TritonPythonModel:
    """Triton Python Model implementing full 'soft' processing pipeline"""

    def initialize(self, args):
        """Initialize the model with soft processing parameters"""
        self.logger = pb_utils.Logger
        self.model_dir = Path(args['model_repository'])
        self.version_dir = self.model_dir / args['model_version']
        
        config_path = self.version_dir / 'config.json'
        with open(config_path) as f:
            self.config = json.load(f)
        self.model_name_to_query = self.config["desired_model"]

        processing_config_path = self.version_dir / f'processing_config_{self.model_name_to_query}.json'
        with open(processing_config_path, 'r') as f:
            self.processing_config = json.load(f)
            
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        self.russian_stopwords = set(nltk.corpus.stopwords.words('russian'))
        self.morph_analyzer = pymorphy3.MorphAnalyzer()
        self.logger.log("Soft processing model initialized")
        self.logger.log(f"Loaded configuration: {json.dumps(self.processing_config, indent=2)}")

    def execute(self, requests):
        """Execute full processing pipeline for batch of requests"""
        responses = []
        
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_texts = [t[0].decode("UTF-8") for t in input_tensor.as_numpy()]
            
            lemmatization_config = self.processing_config['lemmatization']
            cleaning_params = self.processing_config['cleaning']
            
            preprocessed_texts = []
            for text in input_texts:
                cleaned_text = clean(
                    text,
                    fix_unicode=cleaning_params.get('fix_unicode'),
                    to_ascii=cleaning_params.get('to_ascii'),
                    lower=cleaning_params.get('lower'),
                    no_line_breaks=cleaning_params.get('no_line_breaks'),
                    no_urls=cleaning_params.get('no_urls'),
                    no_emails=cleaning_params.get('no_emails'),
                    no_phone_numbers=cleaning_params.get('no_phone_numbers'),
                    no_numbers=cleaning_params.get('no_numbers'),
                    no_digits=cleaning_params.get('no_digits'),
                    no_currency_symbols=cleaning_params.get('no_currency_symbols'),
                    no_punct=cleaning_params.get('no_punct'),
                    replace_with_url=cleaning_params.get('replace_with_url'),
                    replace_with_email=cleaning_params.get('replace_with_email'),
                    replace_with_phone_number=cleaning_params.get('replace_with_phone_number'),
                    replace_with_punct=cleaning_params.get('replace_with_punct'),
                    lang=cleaning_params.get('lang')
                )
                
                tokens = nltk.word_tokenize(cleaned_text, language='russian')
                
                if lemmatization_config['enabled']:
                    filtered_tokens = [
                        token for token in tokens 
                        if token.lower() not in self.russian_stopwords and token.isalpha()
                    ]
                    
                    tokens = [
                        self.morph_analyzer.parse(token)[0].normal_form 
                        for token in filtered_tokens
                    ]
                    
                    if lemmatization_config['keep_only_nouns']:
                        tokens = [
                            t for t in tokens 
                            if 'NOUN' in self.morph_analyzer.parse(t)[0].tag
                        ]
                else:
                    tokens = [t for t in tokens if len(t) > 1]
                
                preprocessed_texts.append(" ".join(tokens))
            
            onnx_input = np.array([[text] for text in preprocessed_texts], dtype=object)
            
            inference_request = pb_utils.InferenceRequest(
                model_name=self.model_name_to_query,
                requested_output_names=['label'],
                inputs=[pb_utils.Tensor("input", onnx_input)]
            )
            inference_response = inference_request.exec()
            
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
            
            output_tensor = pb_utils.get_output_tensor_by_name(inference_response, "label")
            topic = output_tensor.as_numpy()
            
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