import importlib

def load_builder(module_path: str, func_name: str = "build_messages"):
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name, None)
    if not fn:
        raise ImportError(f"{module_path}.{func_name} not found")
    return fn
