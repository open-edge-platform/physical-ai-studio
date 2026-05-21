# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProjectRobotMapper bimanual payload roundtrip."""

from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from repositories.mappers.project_robot_mapper import ProjectRobotMapper
from schemas.robot import RobotType, TrossenBimanualPayload, TrossenBimanualRobot


def _make_bimanual_db_model(robot_type: RobotType):
    model = MagicMock()
    model.id = str(uuid4())
    model.name = "Bimanual Test Robot"
    model.type = str(robot_type)
    model.payload = {
        "connection_string_left": "10.0.0.1",
        "connection_string_right": "10.0.0.2",
        "serial_number": "",
    }
    model.active_calibration_id = None
    model.created_at = datetime(2026, 1, 1)
    model.updated_at = datetime(2026, 1, 1)
    return model


class TestProjectRobotMapperBimanual:
    @pytest.mark.parametrize(
        "robot_type",
        [
            RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER,
            RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER,
        ],
    )
    def test_from_schema_returns_bimanual_robot(self, robot_type):
        db_model = _make_bimanual_db_model(robot_type)
        result = ProjectRobotMapper.from_schema(db_model)

        assert isinstance(result, TrossenBimanualRobot)
        assert result.type == robot_type
        assert isinstance(result.payload, TrossenBimanualPayload)
        assert result.payload.connection_string_left == "10.0.0.1"
        assert result.payload.connection_string_right == "10.0.0.2"

    @pytest.mark.parametrize(
        "robot_type",
        [
            RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER,
            RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER,
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
                serial_number="SN-BIMAN-001",
            ),
        )

        db_obj = ProjectRobotMapper.to_schema(original)
        # Simulate DB read-back by using a mock with same attributes
        db_model = MagicMock()
        db_model.id = db_obj.id
        db_model.name = db_obj.name
        db_model.type = db_obj.type
        db_model.payload = db_obj.payload
        db_model.active_calibration_id = None
        db_model.created_at = None
        db_model.updated_at = None

        restored = ProjectRobotMapper.from_schema(db_model)

        assert isinstance(restored, TrossenBimanualRobot)
        assert restored.payload.connection_string_left == "192.168.10.1"
        assert restored.payload.connection_string_right == "192.168.10.2"
        assert restored.payload.serial_number == "SN-BIMAN-001"
