"""Plugin management CLI commands."""

import sys

import click

from exceptions import BaseException, ResourceInUseError, ResourceType
from plugins.plugin_manager import PluginManager, find_robot_types_in_use_sync


@click.group(name="plugin")
def plugin() -> None:
    """Robot plugin management commands."""


@plugin.command("list")
def list_plugins() -> None:
    """List available and installed plugins."""
    from db.engine import get_sync_db_session

    manager = PluginManager()
    plugin_info = manager.list_plugins()

    with get_sync_db_session() as session:
        in_use = {
            type_ for robot_types in (manager.robot_types(info.id) for info in plugin_info) for type_ in robot_types
        }
        in_use_types = set(find_robot_types_in_use_sync(session, list(in_use)))

    for info in plugin_info:
        status = "installed" if info.installed else "available"
        version = f" v{info.installed_version}" if info.installed_version else ""
        click.echo(f"[{status}] {info.name} ({info.id}){version}")
        click.echo(f"  {info.description}")
        if info.robots:
            click.echo(f"  Robots: {', '.join(robot.type for robot in info.robots)}")
        else:
            click.echo("  Robots: discovered after install")
        if info.extensions:
            click.echo("  Extensions:")
            for extension in info.extensions:
                extension_status = "installed" if extension.installed else "available"
                extension_version = (
                    f" v{extension.installed_version}" if extension.installed_version else ""
                )
                click.echo(
                    f"    [{extension_status}] {extension.name} ({extension.id}){extension_version}"
                )
        if info.installed:
            in_use_types_for_plugin = [robot.type for robot in info.robots if robot.type in in_use_types]
            if in_use_types_for_plugin:
                click.echo(f"  In use: {', '.join(in_use_types_for_plugin)}")


@plugin.command("install")
@click.argument("plugin_id")
def install_plugin(plugin_id: str) -> None:
    """Install a plugin distribution from the manifest."""
    try:
        PluginManager().install(plugin_id)
    except BaseException as error:
        click.echo(f"✗ {error.message}", err=True)
        sys.exit(1)
    click.echo(f"✓ Installed plugin '{plugin_id}'.")
    click.echo("Restart the server to activate the new robot types.")


@plugin.command("uninstall")
@click.argument("plugin_id")
def uninstall_plugin(plugin_id: str) -> None:
    """Uninstall a plugin distribution (blocked while robots use its types)."""
    from db.engine import get_sync_db_session

    manager = PluginManager()
    try:
        robot_types = manager.robot_types(plugin_id)
        with get_sync_db_session() as session:
            in_use_robot_types = find_robot_types_in_use_sync(session, robot_types)
        if in_use_robot_types:
            raise ResourceInUseError(
                ResourceType.PLUGIN,
                plugin_id,
                message=(
                    f"Cannot uninstall plugin '{plugin_id}': {len(in_use_robot_types)} robot(s) use type(s) "
                    f"{', '.join(in_use_robot_types)}. Delete those robots first."
                ),
            )
        manager.uninstall(plugin_id)
    except BaseException as error:
        click.echo(f"✗ {error.message}", err=True)
        sys.exit(1)
    click.echo(f"✓ Uninstalled plugin '{plugin_id}'.")
    click.echo("Restart the server to activate the change.")
