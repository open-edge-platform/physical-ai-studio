from enum import StrEnum


class DatasetAccessMode(StrEnum):
    READ_ONLY = "read_only"
    RECORDING_MUTATION = "recording_mutation"
