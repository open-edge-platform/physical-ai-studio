import multiprocessing
import sys

def is_frozen():
    """Check if the application is running as a PyInstaller bundle."""
    return getattr(sys, "frozen", False)

if is_frozen():
    print("RUNTIME HOOK DEBUG: Detected frozen application.")
    # Darwin (macOS) requires 'spawn' due to system library threads
    # that cause issues when processes are created via 'fork'.
    try:
        print("RUNTIME HOOK DEBUG: Calling multiprocessing.freeze_support().")
        multiprocessing.freeze_support()
    except RuntimeError as e:
        # Ignore if the start method is already locked by an earlier PyInstaller hook or application code
        print(f"RUNTIME HOOK ERROR: Could not set start method: {e}")
