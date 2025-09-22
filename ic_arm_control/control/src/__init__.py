# Import the compiled module
import importlib.util
import sys
import os

# Determine the correct .so file based on Python version and platform
if sys.platform == "darwin":  # macOS
	python_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}-darwin"
else:  # Linux
	python_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}-{os.uname().machine}-linux-gnu"
print('sys platform', python_version)
module_path = os.path.join(os.path.dirname(__file__), f"usb_class.{python_version}.so")

# If the exact version doesn't exist, try to find any compatible .so file
if not os.path.exists(module_path):
	for file in os.listdir(os.path.dirname(__file__)):
		if file.startswith("usb_class.") and file.endswith(".so"):
			# On macOS, prefer darwin .so files
			if sys.platform == "darwin" and "darwin" in file:
				module_path = os.path.join(os.path.dirname(__file__), file)
				break
			# On Linux, prefer linux .so files
			elif sys.platform != "darwin" and "linux" in file:
				module_path = os.path.join(os.path.dirname(__file__), file)
				break
	# If no platform-specific file found, use any .so file as fallback
	if not os.path.exists(module_path):
		for file in os.listdir(os.path.dirname(__file__)):
			if file.startswith("usb_class.") and file.endswith(".so"):
				module_path = os.path.join(os.path.dirname(__file__), file)
				break

# Load the module with error handling
try:
    spec = importlib.util.spec_from_file_location("usb_class", module_path)
    usb_class_module = importlib.util.module_from_spec(spec)
    sys.modules["usb_class"] = usb_class_module
    spec.loader.exec_module(usb_class_module)
    
    # Import the required classes/functions from the module
    usb_class = usb_class_module.usb_class
    can_value_type = usb_class_module.can_value_type
    print(f"Successfully loaded usb_class from {module_path}")
    
except Exception as e:
    print(f"Warning: Failed to load usb_class module: {e}")
    print("Creating mock classes for development...")
    
    # Create mock classes for development/testing
    class MockUSBClass:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    class MockCanValueType:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    usb_class = MockUSBClass
    can_value_type = MockCanValueType
