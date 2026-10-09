# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Policy package imports must not require optional policy extras."""

import subprocess
import sys
import textwrap


def test_policy_exports_work_without_cosmos_optional_dependencies() -> None:
    """Base installs can import policies; only constructing Cosmos3Model requires Diffusers."""
    script = textwrap.dedent(
        """
        import builtins
        import importlib.util
        import sys
        from typing import get_type_hints

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
        import physicalai.train
        import physicalai.policies as policies
        from physicalai.policies.act.policy import ACT
        from physicalai.policies.cosmos3.config import Cosmos3Config
        from physicalai.policies.cosmos3.model import Cosmos3Model
        from physicalai.policies.cosmos3.policy import Cosmos3
        from physicalai.policies.cosmos3 import Cosmos3Preprocessor
        from physicalai.policies.molmoact2.policy import MolmoAct2
        from physicalai.policies.pi05.policy import Pi05

        assert "diffusers" not in sys.modules
        assert "physicalai.policies.cosmos3.pipeline" not in sys.modules
        assert ACT.__name__ == "ACT"
        assert Pi05.__name__ == "Pi05"
        assert MolmoAct2.__name__ == "MolmoAct2"
        assert Cosmos3Config.__name__ == "Cosmos3Config"
        assert Cosmos3Model.__name__ == "Cosmos3Model"
        assert Cosmos3Preprocessor.__name__ == "Cosmos3Preprocessor"
        assert policies.get_physicalai_policy_class("cosmos3") is Cosmos3
        assert "pipeline" in get_type_hints(Cosmos3.__init__)
        assert policies.get_physicalai_policy_class("act") is ACT
        assert isinstance(policies.get_policy("act"), ACT)

        try:
            Cosmos3Model(Cosmos3Config(embodiment="pusht"))
        except ModuleNotFoundError as error:
            assert error.name == "diffusers"
            assert "physicalai-train[cosmos3]" in str(error)
        else:
            raise AssertionError("constructing Cosmos3Model must require its optional diffusers dependency")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=False, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
