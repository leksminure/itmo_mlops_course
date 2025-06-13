import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


def create_client(url: str = "localhost:8000") -> InferenceServerClient:
    triton_client = InferenceServerClient(url)
    return triton_client
