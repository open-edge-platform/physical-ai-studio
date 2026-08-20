from importlib import import_module

from core.logging.log_config import LogConfig


def test_worker_log_info_references_existing_worker_classes() -> None:
    worker_modules = {
        "TrainingWorker": "workers.training_worker",
        "ModelWorker": "workers.model_worker",
        "RobotControlWorker": "workers.robot_control_worker",
        "DatasetImportWorker": "workers.dataset_import_worker",
    }

    for worker_name in LogConfig.worker_log_info:
        if worker_name is None:
            continue

        module = import_module(worker_modules[worker_name])
        worker_class = getattr(module, worker_name)

        assert worker_name == worker_class.ROLE
