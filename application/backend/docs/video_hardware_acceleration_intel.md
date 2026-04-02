# Enable hardware acceleration for video encoding on Intel devices

Intel supports hardware acceleration for video encoding on Intel devices using **oneVPL**.
pyAV is used as video backend. pyAV uses libav / ffmpeg.

For **QSV / oneVPL**, the right path is:

1. build/install **oneVPL**
2. build **FFmpeg** with `--enable-libvpl`
3. build **PyAV from source** against that FFmpeg

Intel’s FFmpeg guidance says to enable VPL is to simply compile with `--enable-libvpl`, and that this enables the `*_qsv` codecs such as `h264_qsv` and `hevc_qsv`.
It also notes that FFmpeg’s `*_qsv` codecs are implemented on top of VPL.

One important detail: Intel’s `libvpl` repo is only the **dispatcher + headers + samples**. You also need an **implementation** installed, such as `oneVPL-intel-gpu` for newer Intel Xe and newer hardware, or Media SDK for legacy graphics.

### 1) Install build dependencies

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake meson ninja-build pkg-config \
  python3-dev python3-venv python3-pip \
  yasm nasm \
  libdrm-dev libva-dev vainfo
```

### 2) Build and install oneVPL

```bash
git clone https://github.com/intel/libvpl
cd libvpl

export VPL_INSTALL_DIR="$HOME/opt/vpl"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$VPL_INSTALL_DIR"
cmake --build build -j"$(nproc)"
cmake --install build
```

Then export the pkg-config path so FFmpeg can find `vpl.pc`:

```bash
export PKG_CONFIG_PATH="$VPL_INSTALL_DIR/lib/pkgconfig:$VPL_INSTALL_DIR/lib64/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$VPL_INSTALL_DIR/lib:$VPL_INSTALL_DIR/lib64:$LD_LIBRARY_PATH"
```

That environment setup is necessary when `libvpl` is not installed to a standard location.

### 3) Make sure the Intel GPU runtime is installed

This is separate from the dispatcher. Without the runtime/implementation, FFmpeg may build but QSV will fail at runtime. [Intel’s install docs](https://github.com/intel/libvpl/blob/main/INSTALL.md) say the base package alone is not enough and you need an implementation as well.

### 4) Install dependencies for SW encoders
If the hardware accelerator is not available, you might want to fall back on SW encoders.
You can skip this step if you do not want to enable these encoders.

For libopenh264:
```bash
git clone https://github.com/cisco/openh264.git
make -j"$(nproc)"
sudo make install
```

For SVT-AV1:
```bash
git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git
cd SVT-AV1
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
sudo cmake --install build

export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
```

### 5) Build FFmpeg with libvpl enabled

```bash
cd ~
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg

./configure \
  --prefix="$HOME/opt/ffmpeg-vpl" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/opt/vpl/include" \
  --extra-ldflags="-L$HOME/opt/vpl/lib -L$HOME/opt/vpl/lib64" \
  --extra-libs="-lpthread -lm" \
  --enable-libvpl \
  --enable-vaapi \
  --enable-shared \
  --enable-libsvtav1 \
  --enable-libopenh264

make -j"$(nproc)"
make install
```

You can remove these lines if you did skip step 4:
```bash
--enable-libsvtav1 \
--enable-libopenh264
```

After install:

```bash
export PATH="$HOME/opt/ffmpeg-vpl/bin:$PATH"
export PKG_CONFIG_PATH="$HOME/opt/ffmpeg-vpl/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$HOME/opt/ffmpeg-vpl/lib:$LD_LIBRARY_PATH"
```

### 6) Verify FFmpeg sees QSV

```bash
ffmpeg -encoders | grep qsv
ffmpeg -h encoder=h264_qsv
```

You should see `h264_qsv`. Intel documents that `h264_qsv`, `hevc_qsv`, and other `*_qsv` codecs are the FFmpeg-facing names when using VPL-backed QSV.

### 7) Install PyAV from source

Inside your uv environment:

```bash
uv pip uninstall av
uv pip install --no-binary av av
```
