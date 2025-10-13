import os
from pathlib import Path

default_home = os.path.join(os.path.expanduser("~"), ".cache")
GETI_ACTION_HOME = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv(
                "GETI_ACTION_HOME",
                os.path.join(default_home, "geti_action"),
            )
        )
    )
)

GETI_ACTION_DATASETS = GETI_ACTION_HOME / "datasets"