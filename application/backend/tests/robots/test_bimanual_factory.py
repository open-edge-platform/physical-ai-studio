# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for RobotClientFactory bimanual WidowXAI cases."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from robots.widowxai.bimanual_adapter import BimanualWidowXAIAdapter
from schemas.robot import RobotType, TrossenBimanualPayload, TrossenBimanualRobot


def _make_bimanual_robot(mode: str) -> TrossenBimanualRobot:
    robot_type = (
        RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER
        if mode == "follower"
        else RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER
    )
    return TrossenBimanualRobot(
        id=uuid4(),
        name="Bimanual Test Robot",
        type=robot_type,
        payload=TrossenBimanualPayload(
            connection_string_left="192.168.1.10",
            connection_string_right="192.168.1.11",
        ),
    )


class TestRobotClientFactoryBimanual:
    @pytest.fixture()
    def factory(self):
        from robots.robot_client_factory import RobotClientFactory

        manager = MagicMock()
        cal_service = MagicMock()
        return RobotClientFactory(robot_manager=manager, calibration_service=cal_service)

    def _mock_bimanual_robot(self):
        mock_joint_names = ["left_shoulder_pan", "right_shoulder_pan"]
        mock_bimanual = MagicMock()
        mock_bimanual.joint_names = mock_joint_names
        mock_bimanual.is_connected.return_value = False
        return mock_bimanual

    @pytest.mark.parametrize("mode", ["follower", "leader"])
    def test_builds_bimanual_adapter(self, factory, mode):
        robot = _make_bimanual_robot(mode)
        mock_bimanual = self._mock_bimanual_robot()

        with (
            patch("robots.robot_client_factory.WidowXAI") as mock_widowxai,
            patch("robots.robot_client_factory.BimanualWidowXAI", return_value=mock_bimanual),
        ):
            mock_widowxai.return_value = MagicMock()
            import asyncio

            result = asyncio.run(factory.build(robot))

        assert isinstance(result, BimanualWidowXAIAdapter)

    @pytest.mark.parametrize(
        "mode,expected_type",
        [
            ("follower", RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER),
            ("leader", RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER),
        ],
    )
    def test_robot_type_correct(self, factory, mode, expected_type):
        robot = _make_bimanual_robot(mode)
        mock_bimanual = self._mock_bimanual_robot()

        with (
            patch("robots.robot_client_factory.WidowXAI") as mock_widowxai,
            patch("robots.robot_client_factory.BimanualWidowXAI", return_value=mock_bimanual),
        ):
            mock_widowxai.return_value = MagicMock()
            import asyncio

            result = asyncio.run(factory.build(robot))

        assert result.robot_type == expected_type

    def test_two_widowxai_instances_created(self, factory):
        robot = _make_bimanual_robot("follower")
        mock_bimanual = self._mock_bimanual_robot()

        with (
            patch("robots.robot_client_factory.WidowXAI") as mock_widowxai,
            patch("robots.robot_client_factory.BimanualWidowXAI", return_value=mock_bimanual),
        ):
            mock_widowxai.return_value = MagicMock()
            import asyncio

            asyncio.run(factory.build(robot))

        # One call per arm
        assert mock_widowxai.call_count == 2
        calls = mock_widowxai.call_args_list
        ips = {c[1]["ip"] for c in calls}
        assert ips == {"192.168.1.10", "192.168.1.11"}

    def test_bimanual_widowxai_constructed_with_both_arms(self, factory):
        robot = _make_bimanual_robot("follower")
        mock_bimanual = self._mock_bimanual_robot()

        with (
            patch("robots.robot_client_factory.WidowXAI") as mock_widowxai,
            patch("robots.robot_client_factory.BimanualWidowXAI", return_value=mock_bimanual) as mock_bim_cls,
        ):
            left_arm = MagicMock()
            right_arm = MagicMock()
            mock_widowxai.side_effect = [left_arm, right_arm]
            import asyncio

            asyncio.run(factory.build(robot))

        mock_bim_cls.assert_called_once_with(left=left_arm, right=right_arm)
