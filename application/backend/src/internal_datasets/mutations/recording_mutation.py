from loguru import logger

from internal_datasets.dataset_client import DatasetClient, DatasetReopenError


class RecordingNotSavedError(RuntimeError):
    """Finalizing failed, before anything was copied: the dataset is as it was.

    The message says where the recording is.
    """


class RecordingOutcomeUnknownError(RuntimeError):
    """Copying back started but did not end cleanly: the dataset may or may not hold the recording.

    The cache is kept. The message says how far the copy got and where the cache is.
    """


class RecordingCleanupError(RuntimeError):
    """The recording was copied back into the dataset, but its cache could not be removed."""


class UnusedCacheCleanupError(RuntimeError):
    """No episode was accepted, so nothing was copied into the dataset, and the cache could not be removed."""


class UnconfirmedCleanupError(RuntimeError):
    """The cache was copied back after an Accept failed, so what the dataset holds is not confirmed, and the cache
    could not be removed."""


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
    """An episode save was attempted, so the cache may differ from the dataset and is copied back."""
    save_failed: bool = False
    """A save was attempted and failed: what it left in the cache, and so in the dataset, is not confirmed."""

    def __init__(self, source_dataset: DatasetClient, cache_dataset: DatasetClient):
        self.cache_dataset = cache_dataset
        self.source_dataset = source_dataset
        self.cache_dataset.prepare_for_writing()

    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        self.cache_dataset.add_frame(obs, act, task)

    def save_episode(self) -> None:
        """Save current recording buffer as episode."""
        self.has_mutation = True
        try:
            self.cache_dataset.save_episode()
        except Exception:
            self.save_failed = True
            raise

    def discard_buffer(self) -> None:
        """Discard current recording buffer."""
        self.cache_dataset.discard_buffer()

    def teardown(self) -> None:
        """If mutation exists apply and then remove cache.

        Raises:
            RecordingNotSavedError: Finalizing failed, before anything was copied; the message says where the
                recording is.
            RecordingOutcomeUnknownError: Copying back failed part way, or every file was copied but the dataset
                could not be opened again; the cache is kept.
            RecordingCleanupError: The recording is in the dataset, but the cache could not be removed.
            UnusedCacheCleanupError: Nothing was copied (no episode was accepted), and the cache could not be removed.
            UnconfirmedCleanupError: The cache was copied after an Accept failed, and it could not be removed.
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
            except DatasetReopenError as error:
                logger.error(
                    f"The recording was copied back, but the dataset did not open again; cache kept in {cache}"
                )
                raise RecordingOutcomeUnknownError(
                    f"{error}. The recorded episodes are also kept in {cache}"
                ) from error
            except Exception as error:
                # The dataset may be missing, partly copied or complete: the copy did not say how far it got.
                logger.error(f"Copying the recording back stopped part way; its episodes are still in {cache}")
                raise RecordingOutcomeUnknownError(
                    f"Copying back stopped part way ({error}), so the dataset may be incomplete. "
                    f"The recorded episodes are kept in {cache}"
                ) from error
        try:
            self.cache_dataset.delete()
        except Exception as error:
            if not self.has_mutation:
                logger.error(f"Nothing was recorded to copy back, and the cache {cache} could not be removed")
                raise UnusedCacheCleanupError(f"the cache {cache} could not be removed ({error})") from error
            if self.save_failed:
                logger.error(f"Copied back after a failed Accept, and the cache {cache} could not be removed")
                raise UnconfirmedCleanupError(
                    f"an Accept failed while recording, so what it holds is not confirmed, and the cache {cache} "
                    f"could not be removed ({error})"
                ) from error
            logger.error(f"The recording was saved, but its cache {cache} could not be removed")
            raise RecordingCleanupError(f"its cache {cache} could not be removed ({error})") from error
