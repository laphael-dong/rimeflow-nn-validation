import inspect
import json
import os
import sys

import ai_edge_litert
import coremltools as ct
import onnx
from ai_edge_litert.interpreter import Interpreter

model_path = sys.argv[1]
model = onnx.load(model_path, load_external_data=False)

def attempt(function):
    try:
        function()
        return {"exitCode": 0, "errorType": None, "error": None}
    except Exception as error:
        return {"exitCode": 1, "errorType": type(error).__name__, "error": str(error)}

coreml = attempt(lambda: ct.convert(model_path, source="auto", skip_model_load=True))
litert = attempt(lambda: Interpreter(model_path=model_path, num_threads=1))
metadata = {item.key: item.value for item in model.metadata_props}
print(json.dumps({
    "python": sys.version.split()[0],
    "onnxVersion": onnx.__version__,
    "onnxMetadata": dict(sorted(metadata.items())),
    "coremltools": {
        "version": ct.__version__,
        "convertSignature": str(inspect.signature(ct.convert)),
        "acceptedSources": ["auto", "tensorflow", "pytorch", "milinternal"],
        "onnxDirectConversionSupported": False,
        "attempt": coreml,
    },
    "litert": {
        "version": "2.1.6",
        "modulePath": "/".join(os.path.normpath(ai_edge_litert.__file__).split(os.sep)[-2:]),
        "interpreterSignature": str(inspect.signature(Interpreter)),
        "acceptedModelFormat": "TFLite FlatBuffer",
        "onnxConverterModulePresent": False,
        "attempt": litert,
    },
}, sort_keys=True))
