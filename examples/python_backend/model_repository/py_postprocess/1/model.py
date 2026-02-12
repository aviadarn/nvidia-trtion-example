import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            x = pb_utils.get_input_tensor_by_name(request, "OUTPUT0").as_numpy()
            msg = f"score_vector={x.tolist()}".encode("utf-8")
            arr = np.array([[msg]], dtype=object)
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("FINAL_TEXT", arr)]))
        return responses
