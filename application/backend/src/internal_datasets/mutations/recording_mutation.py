from loguru import logger

from internal_datasets.dataset_client import DatasetClient


class RecordingNotSavedError(RuntimeError):
    """Finalizing or copying back failed: the recording is not in the dataset. The message says where it was left."""


class RecordingCleanupError(RuntimeError):
    """The recording was copied back into the dataset, but its cache could not be removed."""


class RecordingMutation:
    """This is a mutation wrapper to handle dataset recording.

    if dataset exists:
        copy dataset to cache
    else create new dataset in cache
        add_episodes as you wish
    finalize or discard
    finalize -> remove original dataset and copy over new and remove cache
    discard -> remove cache
    """

    cache_dataset: DatasetClient
    source_dataset: DatasetClient
    has_mutation: bool = False

    def __init__(self, source_dataset: DatasetClient, cache_dataset: DatasetClient):
        self.cache_dataset = cache_dataset
        self.source_dataset = source_dataset
        self.cache_dataset.prepare_for_writing()

    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        self.cache_dataset.add_frame(obs, act, task)

    def save_episode(self) -> None:
        """Save current recording buffer as episode."""
        self.has_mutation = True
        self.cache_dataset.save_episode()

    def discard_buffer(self) -> None:
        """Discard current recording buffer."""
        self.cache_dataset.discard_buffer()

    def teardown(self) -> None:
        """If mutation exists apply and then remove cache.

        Raises:
            RecordingNotSavedError: Finalizing or copying back failed; the message says where the recording is.
            RecordingCleanupError: The recording is in the dataset, but the cache could not be removed.
        """
        logger.debug(f"Teardown recording mutation. Has mutation {self.has_mutation}")
        cache = getattr(self.cache_dataset, "path", "the cache")
        try:
            self.cache_dataset.finalize()
        except Exception as error:
            # The cache is left as it is: it holds what this recording wrote, none copied back yet.
            logger.error(f"Could not finalize the recording; what it recorded is left in {cache}")
            raise RecordingNotSavedError(
                f"Finalizing failed ({error}). What was recorded is left in {cache}"
            ) from error
        if self.has_mutation:
            try:
                self.source_dataset.overwrite(self.cache_dataset)
            except Exception as error:
                logger.error(f"Could not copy the recording back; its episodes are still in {cache}")
                raise RecordingNotSavedError(
                    f"Copying back failed ({error}). The recorded episodes are left in {cache}"
                ) from error
        try:
            self.cache_dataset.delete()
        except Exception as error:
            logger.error(f"The recording was saved, but its cache {cache} could not be removed")
            raise RecordingCleanupError(f"its cache {cache} could not be removed ({error})") from error
