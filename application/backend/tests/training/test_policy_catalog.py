from unittest.mock import patch

from training.policy_catalog import declared_policy_catalog, is_local_policy_available


def test_missing_definition_is_not_schedulable_without_importing_implementations() -> None:
    with (
        patch("physicalai.policies.list_physicalai_policies", return_value=["act"]),
        patch("physicalai.policies.get_physicalai_policy_class") as load_class,
    ):
        assert not is_local_policy_available("removed_plugin_policy")
    load_class.assert_not_called()


def test_missing_dependencies_prevent_local_scheduling_without_loading_a_policy() -> None:
    with (
        patch("physicalai.policies.list_physicalai_policies", return_value=["pi05"]),
        patch("training.policy_catalog.missing_policy_dependencies", return_value=["missing-package"]),
    ):
        assert not is_local_policy_available("pi05")


def test_installed_policy_is_schedulable() -> None:
    with (
        patch("physicalai.policies.list_physicalai_policies", return_value=["act"]),
        patch("training.policy_catalog.missing_policy_dependencies", return_value=[]),
    ):
        assert is_local_policy_available("act")


def test_declared_catalog_accepts_policy_ids_unknown_to_this_branch() -> None:
    catalog = declared_policy_catalog({"future_policy": ["torch"]})
    assert catalog.available[0].id == "future_policy"
    assert catalog.backends == {"future_policy": ["torch"]}
