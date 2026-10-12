"""QEV 0.2.1 typed decisions on onnxruntime + numpy. ``from qev_decide import Decider, choice, noul, score``."""
from .protocol import Question, choice, noul, score

__all__ = ["DEFAULT_SEQ_LEN", "Decider", "Question", "add_device_args", "choice", "default_model_dir", "gpu_providers", "noul",
           "providers_from_args", "score", "select_providers"]
__version__ = "0.1.0"

_LAZY = ("Decider", "DEFAULT_SEQ_LEN", "add_device_args", "default_model_dir", "gpu_providers", "providers_from_args",
         "select_providers")


def __getattr__(name):          # the runtime (onnxruntime) loads on first use; the protocol alone needs only numpy
    if name in _LAZY:
        from . import runtime
        return getattr(runtime, name)
    raise AttributeError(name)
