# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export capability declaration for Patch Policy."""

from physicalai.export import ExportablePolicyMixin, ExportBackend


class PatchPolicyExportMixin(ExportablePolicyMixin):
    """Patch Policy export configuration using the shared checkpoint exporter."""

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Return the export backends supported by Patch Policy.

        Returns:
            No backends until export support has been validated.
        """
        return []
