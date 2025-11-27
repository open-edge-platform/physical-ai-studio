# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform

system = platform.system()
print("Setup Hook: Detected operating system:", system)

if system == "Windows":
    try:
        from winrt.windows.storage import ApplicationData

        app_data_folder = ApplicationData.current.local_folder.path
        print("Setup Hook: Using WinRT ApplicationData local folder:", app_data_folder)
        os.environ["DB_DATA_DIR"] = app_data_folder

        print("Setup Hook: Writing log to:", app_data_folder)
        os.environ["LOGS_DIR"] = app_data_folder
    except ImportError:
        print("Setup Hook: WinRT module not found. Make sure to include it in the PyInstaller build.")
    except OSError:
        print("Setup Hook: Application doesn't run in a UWP context; skipping WinRT local folder setup.")
