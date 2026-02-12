import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            x = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            y = x * 2.0
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT0", y)]))
        return responses
