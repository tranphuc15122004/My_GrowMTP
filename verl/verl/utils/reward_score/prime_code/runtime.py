"""Minimal Python module construction used by the code reward evaluator."""
import types
import sys

class RuntimeModule:
    @staticmethod
    def from_string(name, docstring="", source=""):
        module = types.ModuleType(name, docstring)
        sys.modules[name] = module
        exec(compile(source, "<generated-solution>", "exec"), module.__dict__)
        return module
