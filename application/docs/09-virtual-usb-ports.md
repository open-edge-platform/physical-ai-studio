# Virtual USB Ports

Use this guide when your robot is connected by USB to a different machine (for example, a NUC), but you run Physical AI Studio on your workstation.

You can forward the robot serial device over TCP with `socat` and expose it locally as a virtual serial port (for example, `/dev/ttyVACM0`). Studio then discovers it like a normal USB serial device.

The serial settings in this document are tuned for SO101 (including the `b1000000` baud rate). If you use a different robot or control board, update the device path and serial parameters to match your hardware requirements.

## Architecture

```text
Robot USB <-> NUC  <------> Workstation
                    TCP       |
                              |
                       /dev/ttyVACM0 (socat PTY)
                              |
                      Physical AI Studio
```

## 1. Start serial-to-TCP forwarding on the robot-side machine

Run on the NUC (or other machine physically connected to the robot):

```bash
socat TCP-LISTEN:7001,reuseaddr,fork FILE:/dev/ttyACM0,raw,echo=0,b1000000
```

`TCP-LISTEN` is unencrypted and unauthenticated. Use only on a trusted network, or tunnel this traffic through SSH/VPN.

| Flag                | Purpose                                                                            |
|---------------------|------------------------------------------------------------------------------------|
| `TCP-LISTEN:7001`   | Listen on TCP port 7001.                                                           |
| `reuseaddr`         | Allow quick restart of the process.                                                |
| `fork`              | Handle multiple client connections.                                                |
| `FILE:/dev/ttyACM0` | Robot serial device on the NUC.                                                    |
| `raw,echo=0`        | Raw mode, no local echo.                                                           |
| `b1000000`          | SO101 default: set baud rate to 1 Mbps (required for Feetech motor communication). |

If your robot platform uses a different baud rate or serial profile, replace `b1000000` with the correct value for that platform.

## 2. Create a local virtual serial port on the workstation

Run on the workstation where Studio is running:

```bash
sudo socat \
  PTY,link=/dev/ttyVACM0,raw,echo=0,user=$(id -un),group=$(id -gn),mode=660 \
  TCP:192.168.2.8:7001
```

Replace `192.168.2.8` with your NUC IP.

| Flag                     | Purpose                                     |
|--------------------------|---------------------------------------------|
| `PTY,link=/dev/ttyVACM0` | Create a local PTY at a stable path.        |
| `user,group,mode`        | Grant backend process access to the PTY.    |
| `TCP:192.168.2.8:7001`   | Connect to the robot-side `socat` listener. |

## 3. Verify Studio can detect the virtual port

Check backend serial devices:

```bash
curl http://localhost:7860/api/hardware/serial_devices | jq
```

Expected output includes:

```json
{
  "connection_string": "/dev/ttyVACM0",
  "serial_number": null
}
```

If it is missing, verify the symlink exists:

```bash
ls -la /dev/ttyVACM0
```

Then verify pyserial enumeration from `application/backend`:

```bash
uv run python3 -c "
from serial.tools import list_ports
for p in list_ports.comports():
    print(p.device, p.pid)
"
```

If you run Studio with Docker Compose, also ensure the container can access the virtual device.
Add these mounts to your service in `application/docker/docker-compose.yaml`:

```yaml
volumes:
  - /dev/ttyVACM0:/dev/ttyVACM0
  - /dev/pts:/dev/pts
```

Then restart the container:

```bash
docker compose up -d --force-recreate
```

## Troubleshooting

### Permission denied on `/dev/ttyVACM0`

Recreate the local PTY with explicit ownership and mode:

```bash
sudo socat PTY,link=/dev/ttyVACM0,raw,echo=0,user=$(id -un),group=$(id -gn),mode=660 TCP:192.168.2.8:7001
```

### TCP connection fails

Check network reachability and listener status:

```bash
nc -zv 192.168.2.8 7001
```

### Motor handshake fails (for example, missing motor IDs)

Most often this is a baud-rate mismatch. For SO101, confirm the robot-side command includes `b1000000` and verify serial speed:

```bash
stty -F /dev/ttyACM0
# Expected: speed 1000000 baud
```

### End-to-end echo test

On the robot-side machine:

```bash
socat TCP-LISTEN:7001,reuseaddr,fork -,raw,echo=0
```

On the workstation:

```bash
echo hello > /dev/ttyVACM0
```

If `hello` appears on the robot-side terminal, the tunnel is working end to end.

## Next

- Continue with [Environment Setup](./04-environment-setup.md).
