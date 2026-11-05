# Enable hardware acceleration for video encoding on Intel devices

Intel supports hardware acceleration for video encoding using **oneVPL**. pyAV is used
as the video backend. pyAV uses libav / FFmpeg.

For **QSV / oneVPL**, the right path is:

1. have **oneVPL** available (dispatcher + headers, plus a runtime implementation for
   your Intel GPU)
2. have an **FFmpeg built with `--enable-libvpl`**
3. build **PyAV from source** against that FFmpeg

> **Important — PyAV links the *system* FFmpeg, not a bundled one.**
> PyAV v15 builds against the system FFmpeg via `pkg-config` when installed from
> source (`pip install --no-binary av av`). The prebuilt PyPI wheel bundles its own
> FFmpeg that was compiled *without* `--enable-libvpl`/`--enable-vaapi`, so the
> `*_qsv` and `*_vaapi` codecs are **not** available from the wheel even if your
> system has the oneVPL runtime installed. Rebuilding PyAV from source only gives
> hardware codecs if the FFmpeg development libraries it links against were built
> with the right flags.

---

## 0) Check what your FFmpeg already supports

Run the packaged FFmpeg first. If QSV is already built in, you can use the quick path
below; otherwise use the from-source path.

```bash
ffmpeg -encoders | grep qsv
```

- If you see `h264_qsv`, `hevc_qsv`, etc., your system FFmpeg has QSV. Use
  **Path A (fast)**.
- If nothing shows up, your distro FFmpeg was built without `--enable-libvpl`. Use
  **Path B (from source)**.

---

## Path A — Fast path (distro FFmpeg already has QSV)

Some recent distributions ship FFmpeg with `--enable-libvpl` (e.g. Debian 13
“trixie”, which also provides `libvpl-dev`). In that case you only need the FFmpeg
development packages plus a source rebuild of PyAV.

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  pkg-config \
  libvpl-dev \
  libavcodec-dev \
  libavformat-dev \
  libavutil-dev \
  libswscale-dev \
  libswresample-dev \
  libavdevice-dev \
  libavfilter-dev
```

Then rebuild PyAV from source inside your uv environment. Pin the version to match
your lockfile (here `av==15.1.0`):

```bash
uv pip uninstall --python .venv/bin/python av
uv pip install --python .venv/bin/python --no-binary av --no-cache av==15.1.0
```

Verify the codecs are now visible to PyAV:

```bash
python -c "import av; print('h264_qsv' in av.codecs_available)"
```

> The FFmpeg development packages must match the FFmpeg runtime libraries your
> system uses. On Debian 13 the `libav*-dev` packages and the `ffmpeg` binary both
> ship FFmpeg 7.1 (libavcodec 61), so a source-built PyAV links the system
> `libavcodec61` at runtime.

---

## Path B — From source (distro FFmpeg lacks QSV)

Follow this when your distro’s FFmpeg has no `*_qsv` encoders.

One important detail: Intel’s `libvpl` repo is only the **dispatcher + headers +
samples**. You also need an **implementation** installed, such as `oneVPL-intel-gpu`
for newer Intel Xe and newer hardware, or Media SDK for legacy graphics.

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

That environment setup is necessary when `libvpl` is not installed to a standard
location.

### 3) Make sure the Intel GPU runtime is installed

This is separate from the dispatcher. Without the runtime/implementation, FFmpeg may
build but QSV will fail at runtime. [Intel’s install docs](https://github.com/intel/libvpl/blob/main/INSTALL.md)
say the base package alone is not enough and you need an implementation as well.

### 4) Install dependencies for SW encoders

If the hardware accelerator is not available, you might want to fall back on SW
encoders. You can skip this step if you do not want to enable these encoders.

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

You should see `h264_qsv`. Intel documents that `h264_qsv`, `hevc_qsv`, and other
`*_qsv` codecs are the FFmpeg-facing names when using VPL-backed QSV.

### 7) Install PyAV from source against that FFmpeg

Inside your uv environment, with the environment variables from steps 2 and 5
exported **in the same shell**, rebuild PyAV from source. PyAV’s `setup.py` resolves
the FFmpeg libraries through `pkg-config`, so `PKG_CONFIG_PATH` must point at your
custom FFmpeg install. Pin the version to match your lockfile (here `av==15.1.0`):

```bash
uv pip uninstall --python .venv/bin/python av
uv pip install --python .venv/bin/python --no-binary av --no-cache av==15.1.0
```

Verify the codecs are visible to PyAV:

```bash
python -c "import av; print('h264_qsv' in av.codecs_available)"
```

---

## Recording and training

These steps configure the encoder used when **recording new datasets**. They
do not change model training: training decodes the videos already stored in a
dataset and is independent of the encoder selected for future recordings.
