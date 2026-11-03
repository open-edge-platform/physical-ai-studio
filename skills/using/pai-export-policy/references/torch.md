# Torch Export Notes

- Use Torch export primarily for development/debugging unless a Runtime adapter is available in the active environment.
- Keep Torch artifacts clearly identified in metadata so Runtime does not confuse them with ONNX/OpenVINO exports.
- Do not imply customer deployment support unless the matching Runtime loader is installed and documented.
