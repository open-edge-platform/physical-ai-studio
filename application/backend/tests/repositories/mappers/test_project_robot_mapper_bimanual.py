# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProjectRobotMapper bimanual payload roundtrip."""

from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from physicalai_studio_plugin import RobotCatalogDefinition
from pydantic import BaseModel

from repositories.mappers.project_robot_mapper import ProjectRobotMapper
from robots.catalog.registry import RobotCatalogRegistry
from robots.catalog.widowxai import TrossenBimanualPayload, TrossenBimanualRobot
from schemas.robot import UnavailableRobot


def _make_bimanual_db_model(robot_type: str):
    model = MagicMock()
    model.id = str(uuid4())
    model.name = "Bimanual Test Robot"
    model.type = str(robot_type)
    model.payload = {
        "connection_string_left": "10.0.0.1",
        "connection_string_right": "10.0.0.2",
    }
    model.created_at = datetime(2026, 1, 1)
    model.updated_at = datetime(2026, 1, 1)
    return model


class TestProjectRobotMapperBimanual:
    @pytest.mark.parametrize(
        "robot_type",
        [
            "Trossen_Bimanual_WidowXAI_Follower",
            "Trossen_Bimanual_WidowXAI_Leader",
        ],
    )
    def test_from_schema_returns_bimanual_robot(self, robot_type):
        db_model = _make_bimanual_db_model(robot_type)
        result = ProjectRobotMapper.from_schema(db_model, RobotCatalogRegistry())

        assert result.type == robot_type
        assert isinstance(result.payload, TrossenBimanualPayload)
        assert result.payload.connection_string_left == "10.0.0.1"
        assert result.payload.connection_string_right == "10.0.0.2"

    @pytest.mark.parametrize(
        "robot_type",
        [
            "Trossen_Bimanual_WidowXAI_Follower",
            "Trossen_Bimanual_WidowXAI_Leader",
        ],
    )
    def test_roundtrip_to_schema_and_back(self, robot_type):
        """to_schema then from_schema should preserve all payload fields."""
        original = TrossenBimanualRobot(
            id=uuid4(),
            name="Roundtrip Robot",
            type=robot_type,
            payload=TrossenBimanualPayload(
                connection_string_left="192.168.10.1",
                connection_string_right="192.168.10.2",
            ),
        )

        db_obj = ProjectRobotMapper.to_schema(original)
        # Simulate DB read-back by using a mock with same attributes
        db_model = MagicMock()
        db_model.id = db_obj.id
        db_model.name = db_obj.name
        db_model.type = db_obj.type
        db_model.payload = db_obj.payload
        db_model.created_at = None
        db_model.updated_at = None

        restored = ProjectRobotMapper.from_schema(db_model, RobotCatalogRegistry())

        assert restored.payload.connection_string_left == "192.168.10.1"
        assert restored.payload.connection_string_right == "192.168.10.2"

    def test_from_schema_preserves_robot_from_unavailable_plugin(self):
        db_model = _make_bimanual_db_model("MuJoCo_SO101_Follower")

        result = ProjectRobotMapper.from_schema(db_model, RobotCatalogRegistry())

        assert isinstance(result, UnavailableRobot)
        assert result.type == "MuJoCo_SO101_Follower"
        assert result.unavailable is True
        assert result.payload == db_model.payload

    def test_from_schema_uses_the_injected_registry_adapter(self):
        class PluginPayload(BaseModel):
            connection_string: str

        registry = RobotCatalogRegistry()
        registry.register_robot(
            RobotCatalogDefinition(
                type="Test_Plugin_Robot",
                display_name="Test Plugin Robot",
                role="follower",
                robot_payload=PluginPayload,
            )
        )
        db_model = _make_bimanual_db_model("Test_Plugin_Robot")
        db_model.payload = {"connection_string": "test://robot"}

        result = ProjectRobotMapper.from_schema(db_model, registry)

        assert result.type == "Test_Plugin_Robot"
        assert result.payload.connection_string == "test://robot"
