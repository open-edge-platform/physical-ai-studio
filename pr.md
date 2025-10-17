# Pull Request

## Description
Add a comprehensive VideoLogger callback for PyTorch Lightning training that captures and saves video frames from observations during training, validation, and testing phases. The callback supports multiple camera formats, device compatibility (CPU/CUDA/MPS/XPU), and configurable video output settings.

## Type of Change
- [x] ✨ feat: A new feature
- [ ] 🐛 fix: A bug fix
- [ ] 📚 docs: Documentation only changes
- [ ] 🎨 style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- [ ] ♻️ refactor: A code change that neither fixes a bug nor adds a feature
- [ ] ⚡ perf: A code change that improves performance
- [ ] 🧪 test: Adding missing tests or correcting existing tests
- [ ] 📦 build: Changes that affect the build system or external dependencies
- [ ] 👷 ci: Changes to our CI configuration files and scripts
- [ ] 🔧 chore: Other changes that don't modify src or test files
- [ ] ⏪ revert: Reverts a previous commit

## Related Issues
<!-- If there were any related issues, they would be listed here -->

## Changes Made
- Added `VideoLogger` callback class in `getiaction.train.callbacks`
- Implemented video capture from `Observation` batches during training/validation
- Added support for multiple image formats (tensor, dict of tensors)
- Implemented device-agnostic processing (CPU, CUDA, MPS, XPU)
- Added configurable video output settings (FPS, batch intervals, max videos per phase)
- Organized video files by phase (train/val/test) with epoch and batch naming
- Added comprehensive test suite with 12 test cases covering all functionality
- Updated module exports to include VideoLogger in train package

## Key Features
- **External injection**: Easily add to any Lightning trainer via callbacks parameter
- **Observation compatibility**: Works seamlessly with GetiAction's Observation dataclass
- **Multi-camera support**: Handles both single tensors and camera dictionaries
- **Device agnostic**: Automatically handles tensors on different devices
- **Configurable**: Customizable FPS, logging frequency, and phase selection
- **Organized output**: Structured directory layout with descriptive filenames

## Usage Example
```python
from getiaction.train import VideoLogger

# Add to trainer
video_logger = VideoLogger(
    output_dir="./videos",
    fps=30,
    phases=["train", "val"]
)
trainer = Trainer(callbacks=[video_logger])
```

## Testing
- 12 comprehensive test cases covering all functionality
- Device compatibility tests for CPU, CUDA, MPS, and XPU
- Format compatibility tests for different tensor shapes
- Phase filtering and video limit tests
- All tests passing ✅

## Additional Notes
- Follows PyTorch Lightning callback conventions
- Inspired by lerobot's video logging patterns
- Designed for easy integration with existing GetiAction training workflows
- Placeholder implementation for gym rollout video capture (future enhancement)

## Breaking Changes
None - this is a new feature that doesn't affect existing functionality.

## Deployment Notes
- Requires `imageio` dependency (already in project requirements)
- Videos are saved as MP4 files with configurable quality settings
- Output directory is created automatically if it doesn't exist