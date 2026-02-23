import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("System Path:")
for p in sys.path:
    print(f"  {p}")

print("\n--- Attempting to import rfdetr ---")
try:
    import rfdetr
    print(f"rfdetr imported successfully. File: {rfdetr.__file__}")
    print(f"rfdetr dir: {dir(rfdetr)}")
except ImportError as e:
    print(f"Failed to import rfdetr: {e}")
except Exception as e:
    print(f"An error occurred during import: {e}")

print("\n--- Attempting to import RFDETRMedium from rfdetr ---")
try:
    from rfdetr import RFDETRMedium
    print("Successfully imported RFDETRMedium")
except ImportError as e:
    print(f"Failed to import RFDETRMedium: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
