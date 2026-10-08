# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Policy package imports must not require optional policy extras."""

import subprocess
import sys
import textwrap


def test_act_policy_import_does_not_import_optional_cosmos_dependency() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import importlib.util
        import sys

        real_find_spec = importlib.util.find_spec
        real_import = builtins.__import__

        def find_spec_without_diffusers(name, *args, **kwargs):
            if name == "diffusers" or name.startswith("diffusers."):
                return None
            return real_find_spec(name, *args, **kwargs)

        def block_diffusers(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "diffusers" or name.startswith("diffusers."):
                raise ModuleNotFoundError("diffusers is intentionally unavailable", name="diffusers")
            return real_import(name, globals, locals, fromlist, level)

        importlib.util.find_spec = find_spec_without_diffusers
        builtins.__import__ = block_diffusers

        import physicalai.data
        import physicalai.policies as policies
        assert "physicalai.policies.cosmos3" not in sys.modules
        assert policies.ACT.__name__ == "ACT"
        assert policies.Pi05.__name__ == "Pi05"
        assert policies.MolmoAct2.__name__ == "MolmoAct2"
        from physicalai.policies.cosmos3 import Cosmos3Config, Cosmos3Preprocessor
        assert Cosmos3Config.__name__ == "Cosmos3Config"
        assert Cosmos3Preprocessor.__name__ == "Cosmos3Preprocessor"
        assert policies.get_physicalai_policy_class("act") is policies.ACT
        assert isinstance(policies.get_policy("act"), policies.ACT)

        try:
            policies.get_physicalai_policy_class("cosmos3")
        except ModuleNotFoundError as error:
            assert error.name == "diffusers"
        else:
            raise AssertionError("loading Cosmos3 should require its optional diffusers dependency")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=False, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
