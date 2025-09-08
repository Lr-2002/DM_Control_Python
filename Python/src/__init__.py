# Import the compiled module
import importlib.util
import sys
import os

# Determine the correct .so file based on Python version
python_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}-{os.uname().machine}-linux-gnu"
module_path = os.path.join(os.path.dirname(__file__), f"usb_class.{python_version}.so")

# If the exact version doesn't exist, try to find any .so file
if not os.path.exists(module_path):
    for file in os.listdir(os.path.dirname(__file__)):
        if file.startswith("usb_class.") and file.endswith(".so"):
            module_path = os.path.join(os.path.dirname(__file__), file)
            break

# Load the module
spec = importlib.util.spec_from_file_location("usb_class", module_path)
usb_class_module = importlib.util.module_from_spec(spec)
sys.modules["usb_class"] = usb_class_module
spec.loader.exec_module(usb_class_module)

# Import the required classes/functions from the module
usb_class = usb_class_module.usb_class
can_value_type = usb_class_module.can_value_type
