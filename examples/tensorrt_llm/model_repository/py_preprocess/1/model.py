import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_TEXT")
            raw = text_tensor.as_numpy()[0][0].decode("utf-8")
            vec = np.array([[float(len(raw)), 1.0, 2.0, 3.0]], dtype=np.float32)
            out = pb_utils.Tensor("INPUT0", vec)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
