import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            raw_text = pb_utils.get_input_tensor_by_name(request, "RAW_TEXT")
            batch_size = raw_text.as_numpy().shape[0]
            token_ids = np.zeros((batch_size, 1), dtype=np.int32)
            output_tensor = pb_utils.Tensor("TOKEN_IDS", token_ids)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses
