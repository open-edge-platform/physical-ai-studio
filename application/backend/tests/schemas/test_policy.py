from unittest.mock import patch

from schemas.policy import AvailablePolicyReference, UnavailablePolicy, UnknownPolicyAvailability
from training.policy_catalog import AvailablePolicy, PolicyCatalog, declared_policy_catalog, readable_policy


def test_missing_definition_preserves_policy_id_without_importing_implementations() -> None:
    with (
        patch("physicalai.policies.list_physicalai_policies", return_value=["act"]),
        patch("physicalai.policies.get_physicalai_policy_class") as load_class,
    ):
        reference = readable_policy("removed_plugin_policy")
    assert isinstance(reference, UnavailablePolicy)
    assert reference.id == "removed_plugin_policy"
    assert reference.unavailable is True
    load_class.assert_not_called()


def test_missing_dependencies_are_unavailable_without_loading_a_policy() -> None:
    with (
        patch("physicalai.policies.list_physicalai_policies", return_value=["pi05"]),
        patch("training.policy_catalog.missing_policy_dependencies", return_value=["missing-package"]),
    ):
        reference = readable_policy("pi05")
    assert isinstance(reference, UnavailablePolicy)
    assert "missing-package" in reference.reason


def test_remote_policy_is_unknown_without_a_verified_target_catalog() -> None:
    with patch("physicalai.policies.list_physicalai_policies") as local_inventory:
        reference = readable_policy("remote_only_policy", remote=True)
    assert isinstance(reference, UnknownPolicyAvailability)
    local_inventory.assert_not_called()


def test_verified_remote_catalog_does_not_depend_on_local_policy_availability() -> None:
    catalog = PolicyCatalog(
        available=[AvailablePolicy(id="remote_only_policy", name="Remote", description="", min_vram=None)],
        backends={},
    )
    assert isinstance(readable_policy("remote_only_policy", catalog, remote=True), AvailablePolicyReference)
    assert isinstance(readable_policy("removed_policy", catalog, remote=True), UnavailablePolicy)


def test_declared_catalog_accepts_policy_ids_unknown_to_this_branch() -> None:
    catalog = declared_policy_catalog({"future_policy": ["torch"]})
    assert catalog.available[0].id == "future_policy"
    assert catalog.backends == {"future_policy": ["torch"]}
