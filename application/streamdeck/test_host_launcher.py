"""Host launcher tests never register handlers on the machine running the suite."""

import json
import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import host_launcher as launcher


class HostLauncherTests(unittest.TestCase):
    def test_links_cannot_change_backend_or_supply_commands(self):
        for link in (
            "https://example.com",
            launcher.LINK + "?url=http://attacker",
            launcher.LINK + "#x",
            launcher.LINK + "/extra",
            launcher.LINK + ' --url "http://attacker"',
        ):
            with self.subTest(link=link), self.assertRaises(ValueError):
                launcher.open_link(link)

    def test_backend_requires_an_explicit_origin(self):
        self.assertEqual(launcher.validate_backend("https://studio.example:8443/"), "https://studio.example:8443")
        self.assertEqual(launcher.validate_backend("http://[::1]:7860"), "http://[::1]:7860")
        for url in (
            "file:///tmp/x",
            "http://user:pass@localhost",
            "http://localhost/api",
            "http://host?x=1",
            "http://host#x",
            "http://host:invalid",
            "http://host\n",
            "http://[invalid",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                launcher.validate_backend(url)

    def test_install_copies_scripts_and_pins_backend_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "installed"
            with (
                patch.object(launcher, "install_dir", return_value=root),
                patch.object(launcher.shutil, "which", return_value="/tools/uv"),
                patch.object(launcher, "register") as register,
            ):
                self.assertEqual(launcher.install("http://my-studio:7860"), root)
                self.assertTrue((root / "streamdeck_bridge.py").is_file())
                self.assertTrue((root / "dictation.py").is_file())
                self.assertTrue((root / "host_launcher.py").is_file())
                self.assertEqual(
                    json.loads((root / "config.json").read_text()), {"uv": "/tools/uv", "url": "http://my-studio:7860"}
                )
                register.assert_called_once_with(root, "/tools/uv")

    @unittest.skipIf(launcher.sys.platform == "win32", "POSIX flock test")
    def test_open_uses_saved_config_and_duplicate_launch_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "config.json").write_text(json.dumps({"uv": "/tools/uv", "url": "http://localhost:7860"}))
            with (
                patch.object(launcher, "__file__", str(root / "host_launcher.py")),
                patch.object(launcher.subprocess, "run") as run,
            ):
                run.side_effect = lambda *args, **kwargs: launcher.open_link(launcher.LINK)
                launcher.open_link(launcher.LINK + "/")
                self.assertEqual(run.call_count, 1)
                self.assertEqual(
                    run.call_args.args[0],
                    [
                        "/tools/uv",
                        "run",
                        "--no-project",
                        str(root / "streamdeck_bridge.py"),
                        "--url",
                        "http://localhost:7860",
                    ],
                )
                self.assertNotIn("shell", run.call_args.kwargs)

    def test_linux_registration_quotes_paths_and_leaves_uri_as_one_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data" / "physicalai" / "streamdeck"
            with patch.object(launcher.sys, "platform", "linux"), patch.object(launcher.subprocess, "run") as run:
                launcher.register(root, '/tools with spaces/uv%"')
            entry = (root.parents[1] / "applications" / launcher.DESKTOP_NAME).read_text()
            self.assertIn(r'Exec="/tools with spaces/uv%%\\"" "run"', entry)
            self.assertTrue(entry.endswith('"--open" %u\n'))
            run.assert_called_once_with(
                [
                    "xdg-mime",
                    "default",
                    launcher.DESKTOP_NAME,
                    "x-scheme-handler/physicalai-streamdeck",
                ],
                check=True,
            )

    def test_desktop_escaping_covers_both_string_and_exec_quoting(self):
        self.assertEqual(launcher.desktop_argument(r"a\b$c"), r'"a\\\\b\\$c"')

    def test_windows_registration_uses_user_registry_and_quotes_uri(self):
        registry = MagicMock()
        registry.HKEY_CURRENT_USER = "user"
        with patch.object(launcher.sys, "platform", "win32"), patch.dict("sys.modules", winreg=registry):
            launcher.register(Path("installed"), "C:/Program Files/uv.exe")
        self.assertTrue(all(call.args[0] == "user" for call in registry.CreateKey.call_args_list))
        command = registry.SetValueEx.call_args.args[-1]
        self.assertTrue(command.startswith('"C:/Program Files/uv.exe"'))
        self.assertTrue(command.endswith(' --open "%1"'))

    def test_mac_registration_escapes_uri_and_sets_url_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            app = home / "Applications" / f"{launcher.APP_NAME}.app"

            def compile_app(command, **kwargs):
                if command[0] == "/usr/bin/osacompile":
                    info = app / "Contents" / "Info.plist"
                    info.parent.mkdir(parents=True)
                    info.write_bytes(plistlib.dumps({"CFBundleName": launcher.APP_NAME}))

            with (
                patch.object(launcher.sys, "platform", "darwin"),
                patch.object(launcher.Path, "home", return_value=home),
                patch.object(launcher.subprocess, "run", side_effect=compile_app) as run,
            ):
                launcher.register(home / "data", "/tools/uv")
            self.assertIn("quoted form of launchURL", run.call_args_list[0].args[0][-1])
            metadata = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
            self.assertEqual(metadata["CFBundleURLTypes"][0]["CFBundleURLSchemes"], [launcher.SCHEME])
            self.assertTrue(metadata["LSUIElement"])
            self.assertIn("Dictate task prompts locally", metadata["NSMicrophoneUsageDescription"])
            self.assertIn("-f", run.call_args.args[0])

    def test_linux_uninstall_removes_only_known_files_and_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data" / "physicalai" / "streamdeck"
            root.mkdir(parents=True)
            for name in ("host_launcher.py", "streamdeck_bridge.py", "dictation.py", "config.json", "helper.log"):
                (root / name).write_text("saved")
            entry = root.parents[1] / "applications" / launcher.DESKTOP_NAME
            entry.parent.mkdir()
            entry.write_text("entry")
            with (
                patch.object(launcher.sys, "platform", "linux"),
                patch.object(launcher, "install_dir", return_value=root),
            ):
                launcher.uninstall()
            self.assertEqual(list(root.iterdir()), [root / "helper.log"])
            self.assertFalse(entry.exists())


if __name__ == "__main__":
    unittest.main()
