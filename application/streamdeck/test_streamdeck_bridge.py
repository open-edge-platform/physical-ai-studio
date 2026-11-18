"""Hardware-free checks; see README.md for the test command."""

import json
import unittest
from unittest.mock import Mock, patch

import httpx
from streamdeck_bridge import (
    DISABLED,
    AttachmentChanged,
    DeckDisplay,
    ManagedInferenceBridge,
    ManagedRecordingBridge,
    RecordingBridge,
    main,
    run_deck,
    run_managed,
)


class RecordingBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "environment": {"name": "Bench"},
            "dataset": {"is_recording": False, "episodes_recorded": 3},
        }
        self.commands = []
        self.offline = False
        self.timeout = False
        self.ack = {"ok": True}

        def respond(request: httpx.Request) -> httpx.Response:
            if self.offline:
                raise httpx.ConnectError("offline", request=request)
            if request.method == "GET":
                return httpx.Response(200, json=self.state)
            self.commands.append(json.loads(request.content))
            if self.timeout:
                raise httpx.ReadTimeout("response lost", request=request)
            return httpx.Response(200, json=self.ack)

        self.client = httpx.Client(base_url="http://studio", transport=httpx.MockTransport(respond))
        self.addCleanup(self.client.close)
        self.bridge = RecordingBridge(self.client, "chosen-session", "pick blocks")
        self.bridge.refresh()

    def test_explicit_commands_require_fresh_server_state(self) -> None:
        self.bridge.press(0, 10)
        self.assertEqual(self.commands[0]["command"], "start_episode")
        self.assertEqual(self.commands[0]["task"], "pick blocks")
        self.assertTrue(self.commands[0]["request_id"])
        self.assertIsNone(self.bridge.state)
        self.bridge.press(0, 11)
        self.assertEqual(len(self.commands), 1)
        self.state["dataset"]["is_recording"] = True
        self.bridge.refresh()
        self.bridge.press(0, 12)  # start must not become a toggle
        self.assertEqual(len(self.commands), 1)
        self.bridge.press(1, 13)
        self.assertEqual(self.commands[-1]["command"], "save_episode")
        self.assertNotEqual(self.commands[0]["request_id"], self.commands[1]["request_id"])

    def test_no_dataset_or_environment_disables_start(self) -> None:
        for missing in ("environment", "dataset"):
            with self.subTest(missing=missing):
                original = self.state[missing]
                self.state[missing] = None
                self.bridge.refresh()
                for key in range(3):
                    self.bridge.press(key, 10)
                self.assertEqual(self.commands, [])
                self.assertEqual(self.bridge.buttons(10)[0][1], DISABLED)
                self.state[missing] = original

    def test_discard_requires_confirmation_and_expires(self) -> None:
        self.state["dataset"]["is_recording"] = True
        self.bridge.refresh()
        self.bridge.press(2, 10)
        self.assertEqual(self.commands, [])
        self.assertIn("CONFIRM", self.bridge.buttons(11)[2][0])
        self.bridge.press(2, 12)  # expired, rearm instead
        self.assertEqual(self.commands, [])
        self.bridge.press(2, 13)
        self.assertEqual(self.commands[0]["command"], "discard_episode")

    def test_disconnect_disarms_and_recovery_uses_actual_state(self) -> None:
        self.state["dataset"]["is_recording"] = True
        self.bridge.refresh()
        self.bridge.press(2, 10)
        self.offline = True
        with self.assertRaises(httpx.ConnectError):
            self.bridge.refresh()
        self.bridge.press(2, 11)
        self.assertEqual(self.commands, [])
        self.assertEqual(self.bridge.discard_until, 0)
        self.assertEqual(self.bridge.buttons(11)[3][0], "OFFLINE")
        self.offline = False
        self.state["dataset"]["is_recording"] = False
        self.state["dataset"]["episodes_recorded"] = 4
        self.bridge.refresh()
        self.assertEqual(self.bridge.buttons(12)[4][0], "SAVED\n4")
        self.assertEqual(self.bridge.buttons(12)[5][0], "CONNECTED")

    def test_ack_failure_is_visible_without_optimistic_state(self) -> None:
        self.ack = {"ok": False, "error": {"message": "disk full"}}
        with self.assertLogs(level="ERROR"):
            self.bridge.press(0, 10)
        self.assertIsNone(self.bridge.state)
        self.bridge.refresh()
        self.assertEqual(self.bridge.buttons(11)[3][0], "READY")
        self.assertEqual(self.bridge.buttons(11)[5][0], "ERROR\nSEE LOG")

    def test_timeout_never_replays_a_command(self) -> None:
        self.timeout = True
        with self.assertRaises(httpx.ReadTimeout):
            self.bridge.press(0, 10)
        self.bridge.press(0, 11)
        self.bridge.refresh()
        self.assertEqual(len(self.commands), 1)
        self.assertIsNotNone(self.bridge.state)

    def test_list_sessions_does_not_probe_usb(self) -> None:
        response = httpx.Response(200, json={"sessions": []}, request=httpx.Request("GET", "http://studio"))
        with (
            patch("sys.argv", ["bridge", "--list-sessions"]),
            patch("streamdeck_bridge.httpx.Client") as client,
            patch("streamdeck_bridge.DeviceManager") as manager,
        ):
            client.return_value.__enter__.return_value.get.return_value = response
            main()
            manager.assert_not_called()

    def test_input_during_network_work_is_dropped(self) -> None:
        deck = Mock()
        deck.key_layout.return_value = (2, 3)
        deck.is_open.return_value = True
        bridge = Mock(state=self.state)
        bridge.buttons.return_value = self.bridge.buttons(10)
        calls = 0

        def refresh() -> None:
            nonlocal calls
            calls += 1
            callback = deck.set_key_callback.call_args.args[0]
            callback(deck, 0, True)  # cannot queue a command while polling
            if calls == 2:
                raise KeyboardInterrupt

        bridge.refresh.side_effect = refresh
        with patch.object(DeckDisplay, "render"), self.assertRaises(KeyboardInterrupt):
            run_deck(deck, bridge)
        bridge.press.assert_not_called()
        deck.set_key_callback.assert_called_with(None)


class ManagedBridgeTests(unittest.TestCase):
    def test_ui_task_sync_detachment_and_failed_heartbeat_disable_input(self) -> None:
        target = {"session_id": "one", "task": "Old task", "connection_id": "connection"}
        desired = dict(target)
        state = {"environment": {}, "dataset": {"is_recording": False, "episodes_recorded": 0}}
        state["environment"] = {"name": "Bench"}
        offline = False
        commands = []

        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/streamdeck/heartbeat":
                if offline:
                    raise httpx.ConnectError("offline", request=request)
                return httpx.Response(200, json={"target": desired})
            if request.method == "GET":
                return httpx.Response(200, json=state)
            commands.append(request)
            return httpx.Response(200, json={"ok": True})

        with httpx.Client(base_url="http://studio", transport=httpx.MockTransport(respond)) as client:
            bridge = ManagedRecordingBridge(client, "helper", target, "Mini")
            desired["task"] = "  "
            bridge.refresh()
            bridge.press(0, 10)
            self.assertEqual(commands, [])
            self.assertEqual(bridge.buttons(10)[0][1], DISABLED)
            desired["task"] = "New task from Studio"
            bridge.refresh()
            bridge.press(0, 11)
            self.assertEqual(json.loads(commands[0].content)["task"], "New task from Studio")
            self.assertEqual(commands[0].headers["X-StreamDeck-Connection"], "connection")
            bridge.refresh()
            bridge.discard_until = 100
            offline = True
            with self.assertRaises(httpx.ConnectError):
                bridge.refresh()
            bridge.press(0, 12)
            self.assertIsNone(bridge.state)
            self.assertEqual(bridge.discard_until, 0)
            offline = False
            bridge.refresh()
            desired["connection_id"] = "replacement"
            with self.assertRaises(AttachmentChanged):
                bridge.refresh()
            bridge.press(0, 13)
            self.assertEqual(len(commands), 1)

    def test_no_arguments_starts_idle_helper_without_claiming_usb(self) -> None:
        with (
            patch("sys.argv", ["bridge"]),
            patch("streamdeck_bridge.heartbeat", return_value={"target": None}),
            patch("streamdeck_bridge.time.sleep", side_effect=KeyboardInterrupt),
            patch("streamdeck_bridge.DeviceManager") as manager,
        ):
            main()
            manager.assert_not_called()

    def test_ui_disconnect_releases_usb_before_waiting_for_another_attachment(self) -> None:
        deck = Mock()
        deck.is_visual.return_value = True
        target = {"session_id": "one", "task": "Pick block", "connection_id": "connection"}
        with (
            patch("streamdeck_bridge.DeviceManager") as manager,
            patch("streamdeck_bridge.heartbeat", side_effect=[{"target": target}, {"target": None}]),
            patch("streamdeck_bridge.run_deck", side_effect=AttachmentChanged),
            patch("streamdeck_bridge.time.sleep", side_effect=KeyboardInterrupt),
        ):
            manager.return_value.enumerate.return_value = [deck]
            with self.assertRaises(KeyboardInterrupt):
                run_managed(Mock())
        deck.open.assert_called_once()
        deck.close.assert_called_once()
        self.assertEqual(deck.reset.call_count, 2)


class InferenceBridgeTests(unittest.TestCase):
    def setUp(self):
        self.target = {"session_id": "one", "task": "Task", "page": "inference", "connection_id": "chosen"}
        self.state = {"environment": {"name": "Bench"}, "model": {"name": "Policy"}, "control": None}
        self.commands = []
        self.offline = False
        self.timeout = False
        self.ack = {"ok": True}

        def respond(request):
            if self.offline:
                raise httpx.ConnectError("offline", request=request)
            if request.url.path.endswith("heartbeat"):
                return httpx.Response(200, json={"target": self.target})
            if request.method == "GET":
                return httpx.Response(200, json=self.state)
            self.commands.append(request)
            if self.timeout:
                raise httpx.ReadTimeout("outcome unknown", request=request)
            return httpx.Response(200, json=self.ack)

        self.client = httpx.Client(base_url="http://studio", transport=httpx.MockTransport(respond))
        self.addCleanup(self.client.close)
        self.bridge = ManagedInferenceBridge(self.client, "helper", self.target, "Mini")
        self.bridge.refresh()

    def test_microphone_delivers_only_a_draft_then_start_and_stop_are_separate(self):
        voice = self.bridge.dictation
        with patch.object(voice, "start") as start:
            self.bridge.press(2, 1)
            start.assert_called_once()
        voice.status = "listening"
        self.assertIn("FINISH", self.bridge.buttons(1)[2][0])
        self.bridge.press(2, 2)
        self.assertTrue(voice.stop.is_set())
        self.bridge.press(0, 3)
        self.assertEqual(self.commands, [])
        voice.status, voice.task = "done", "New spoken task"
        self.bridge.refresh()
        self.assertTrue(self.commands[0].url.path.endswith("dictation"))
        self.assertEqual(json.loads(self.commands[0].content), {"task": "New spoken task"})
        self.bridge.refresh()
        self.assertEqual(len(self.commands), 1)
        self.bridge.press(0, 4)
        self.assertEqual(json.loads(self.commands[1].content)["action"], "start")
        self.assertEqual(self.commands[1].headers["X-StreamDeck-Connection"], "chosen")
        self.assertIsNone(self.bridge.state)
        self.bridge.press(0, 5)
        self.assertEqual(len(self.commands), 2)
        self.state["control"] = {"type": "model"}
        self.bridge.refresh()
        self.bridge.press(1, 6)
        self.assertEqual(json.loads(self.commands[2].content)["action"], "stop")

    def test_nonmodel_control_missing_parts_and_voice_busy_do_not_start_or_take_over(self):
        with patch.object(self.bridge.dictation, "start") as start:
            self.state["control"] = {"type": "teleop"}
            self.bridge.refresh()
            for key in range(3):
                self.bridge.press(key, 1)
            start.assert_not_called()
        self.state["control"] = None
        self.state["model"] = None
        self.bridge.refresh()
        self.bridge.press(0, 1)
        self.assertEqual(self.bridge.buttons(1)[0][1], DISABLED)
        self.assertEqual(self.commands, [])

    def test_detachment_and_connection_failure_cancel_capture_and_drop_late_results(self):
        self.offline = True
        with self.assertRaises(httpx.ConnectError):
            self.bridge.refresh()
        self.assertTrue(self.bridge.dictation.cancelled.is_set())
        self.assertIsNone(self.bridge.state)
        self.offline = False
        self.bridge.dictation.status, self.bridge.dictation.task = "done", "late"
        self.bridge.refresh()
        self.assertEqual(self.commands, [])
        self.target = {**self.target, "connection_id": "new-page"}
        with self.assertRaises(AttachmentChanged):
            self.bridge.refresh()
        self.bridge.press(0, 1)
        self.assertEqual(self.commands, [])

    def test_timeout_never_replays_commands_or_transcripts_and_ack_errors_are_visible(self):
        self.timeout = True
        with self.assertRaises(httpx.ReadTimeout):
            self.bridge.press(0, 1)
        self.bridge.press(0, 2)
        self.bridge.refresh()
        self.assertEqual(len(self.commands), 1)
        self.bridge.dictation.status, self.bridge.dictation.task = "done", "spoken task"
        with self.assertRaises(httpx.ReadTimeout):
            self.bridge.refresh()
        self.bridge.refresh()
        self.assertEqual(len(self.commands), 2)
        self.timeout = False
        self.ack = {"ok": False, "error": {"message": "model unavailable"}}
        with self.assertLogs(level="ERROR"):
            self.bridge.press(0, 3)
        self.assertIn("ERROR", self.bridge.buttons(3)[5][0])
        self.assertIsNone(self.bridge.state)


class DisplayTests(unittest.TestCase):
    def test_mini_and_neo_mapping_and_unchanged_images(self) -> None:
        for columns, expected in ((3, [0, 1, 2, 3, 4, 5]), (4, [0, 1, 2, 4, 5, 6])):
            with self.subTest(columns=columns):
                deck = Mock()
                deck.key_layout.return_value = (2, columns)
                deck.key_image_format.return_value = {
                    "size": (80, 80),
                    "format": "JPEG",
                    "flip": (False, False),
                    "rotation": 0,
                }
                display = DeckDisplay(deck)
                buttons = [("START", DISABLED)] * 6
                display.render(buttons)
                display.render(buttons)
                self.assertEqual([call.args[0] for call in deck.set_key_image.call_args_list], expected)


if __name__ == "__main__":
    unittest.main()
