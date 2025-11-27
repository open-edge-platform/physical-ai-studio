import multiprocessing
import sys

# This hook runs extremely early, before most of your application's imports.

def is_frozen():
    """Check if the application is running as a PyInstaller bundle."""
    return getattr(sys, "frozen", False)

if sys.platform == "darwin":
    print("RUNTIME HOOK DEBUG: Detected macOS in frozen application.")
    # Darwin (macOS) requires 'spawn' due to system library threads
    # that cause issues when processes are created via 'fork'.
    try:
        print("RUNTIME HOOK DEBUG: Calling multiprocessing.freeze_support().")
        multiprocessing.freeze_support()

    except RuntimeError as e:
        # Ignore if the start method is already locked by an earlier PyInstaller hook or application code
        print(f"RUNTIME HOOK ERROR: Could not set start method: {e}")
elif sys.platform == "darwin":
    print("RUNTIME HOOK DEBUG: Detected macOS, but not running in a frozen application. Skipping hook logic.")
else:
    print(f"RUNTIME HOOK DEBUG: Detected platform {sys.platform}. Skipping macOS specific logic.")


# Optional: You can also use this hook to set environment variables
# that might be causing issues with certain libraries like joblib/scikit-learn.
# os.environ['JOBLIB_MULTIPROCESSING'] = '0' # Uncomment if needed