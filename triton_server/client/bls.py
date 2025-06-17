from uuid import uuid4

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from utils import create_client


if __name__ == "__main__":
    message: str = "Привет, ты мне нравишься!"

    message_array = np.array([[message.encode()], [message.encode()]], dtype=np.object_)
    print("----- INPUT ------")
    print(f"INPUT: {message_array.shape}")


    input_0 = InferInput(
        "INPUT",
        list(message_array.shape),
        np_to_triton_dtype(message_array.dtype)
    )
    input_0.set_data_from_numpy(message_array)
    inputs = [input_0]

    triton_client = create_client()

    try:
        resp = triton_client.infer("topic_classification_bls", inputs, request_id=str(uuid4()))
    except InferenceServerException as ex:
        print(ex)
    else:
        print("----- OUTPUT -----")
        result = resp.as_numpy("OUTPUT")
        decoded_output = np.array([[byte.decode('utf-8') for byte in row] for row in result])
        print(f"OUTPUT: {decoded_output.shape}")
        print(decoded_output)
    finally:
        triton_client.close()
