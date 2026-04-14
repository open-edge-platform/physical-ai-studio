# SO-101 Serial Troubleshooting

## Hanging on connect / read (unresponsive script)

### Symptoms
- Script prints `Connecting to SO-101 on /dev/tty.usbmodem...` and hangs
- `Ctrl+C` does not work (before the 50ms timeout fix)
- Works without calibration but hangs with calibration file

### Root Cause
The Feetech SDK's `txRxPacket()` is a blocking call with no default timeout.
If a previous process was force-killed (`kill -9`), the serial port is left in
a dirty state — the USB-serial chip still has corrupted buffered data, and the
OS may not have released the file descriptor.

### Resolution Steps

1. **Kill zombie processes:**
   ```bash
   killall -9 python python3
   ```

2. **Check if anything holds the port:**
   ```bash
   lsof /dev/tty.usbmodem5A7A0156901
   ```

3. **Unplug and replug the USB cable** to reset the USB-serial chip firmware
   and force the OS to release the file descriptor.

4. **Verify the port is back:**
   ```bash
   ls /dev/tty.usbmodem*
   ```

5. **Retry with a safe frequency** (≤100 Hz):
   ```bash
   python read_joints.py --port /dev/tty.usbmodem5A7A0156901 --hz 10
   ```

### Alternative to unplugging
```bash
stty -f /dev/tty.usbmodem5A7A0156901 hupcl
```

### Prevention
- The driver now sets a 50ms packet timeout (`setPacketTimeoutMillis`) so that
  pings and reads fail fast instead of blocking forever.
- Always use `Ctrl+C` to stop scripts cleanly. The `KeyboardInterrupt` handler
  in `read_joints.py` calls `disconnect()` which properly closes the port.
- Do not exceed ~100 Hz for 6 servos on a single UART at 1 Mbaud. 1000 Hz
  saturates the bus.

## Bare connectivity test

If the driver hangs, test the SDK directly:

```python
from scservo_sdk import PortHandler, PacketHandler

ph = PortHandler("/dev/tty.usbmodem5A7A0156901")
ph.openPort()
ph.setBaudRate(1_000_000)
pkt = PacketHandler(0)

for sid in range(1, 7):
    model, res, err = pkt.ping(ph, sid)
    print(f"Servo {sid}: model={model}, result={res}, error={err}")

ph.closePort()
```

If this also hangs → hardware/port issue (unplug-replug). If this works but
the driver doesn't → file an issue with the driver output.
