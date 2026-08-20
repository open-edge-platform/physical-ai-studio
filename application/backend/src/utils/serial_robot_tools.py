from loguru import logger
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo

from schemas import SerialPortInfo


def from_port(port: ListPortInfo) -> SerialPortInfo | None:
    """Detect if the device is a compatible serial robot port."""
    serial_number = getattr(port, "serial_number", None)

    # Ignore internal hardware (e.g. /dev/ttyS0..ttyS31)
    ttys_suffix = port.device.removeprefix("/dev/ttyS")
    if ttys_suffix[:1].isdigit():
        return None

    # The Feetech UART board CH340 has PID 29987
    # Also accept virtual/PTY ports (pid is None) like socat-created devices
    if port.pid is not None and port.pid not in {21971, 29987}:
        logger.debug("Found usb port with unexpected PID, {device}: {pid}", device=port.device, pid=port.pid)

    return SerialPortInfo(connection_string=port.device, serial_number=serial_number or None)


class RobotConnectionManager:
    _all_robots: list[SerialPortInfo]
    available_ports: list[ListPortInfo]

    def __init__(self):
        self.available_ports = list(list_ports.comports())
        self._all_robots = []

    @property
    def robots(self) -> list[SerialPortInfo]:
        return self._all_robots

    async def find_robots(self) -> None:
        self.available_ports = list(list_ports.comports())
        self._all_robots = []

        connected_devices: set[str] = set()
        connected_serials: set[str] = set()

        for port in self.available_ports:
            serial_num = getattr(port, "serial_number", None)
            if port.device in connected_devices or (serial_num and serial_num in connected_serials):
                logger.debug(f"Skipping {port.device}: already connected (or alias).")
                continue

            robot = from_port(port)
            if robot is None:
                continue

            logger.debug(f"Robot created: {robot}")
            self._all_robots.append(robot)
            connected_devices.add(port.device)
            if serial_num:
                connected_serials.add(serial_num)

        if not self._all_robots:
            logger.debug("No robot connected.")
