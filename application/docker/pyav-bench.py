import argparse
import time

import av
import numpy as np


def generate_yuv420p_frames(width, height, num_frames, seed=1234):
    rng = np.random.default_rng(seed)

    frames = []
    for i in range(num_frames):
        # deterministic but changing over time
        y = rng.integers(16, 236, size=(height, width), dtype=np.uint8)
        u = rng.integers(16, 241, size=(height // 2, width // 2), dtype=np.uint8)
        v = rng.integers(16, 241, size=(height // 2, width // 2), dtype=np.uint8)
        frames.append((y, u, v))

    return frames


def generate_nv12_frames(width, height, num_frames, seed=1234):
    rng = np.random.default_rng(seed)

    frames = []
    for i in range(num_frames):
        y = rng.integers(16, 236, size=(height, width), dtype=np.uint8)

        # NV12 has one interleaved UV plane of shape (h/2, w)
        uv = rng.integers(16, 241, size=(height // 2, width), dtype=np.uint8)

        frames.append((y, uv))

    return frames


def video_frame_from_yuv420p(y, u, v):
    h, w = y.shape
    frame = av.VideoFrame(w, h, "yuv420p")
    frame.planes[0].update(y.tobytes())
    frame.planes[1].update(u.tobytes())
    frame.planes[2].update(v.tobytes())
    return frame


def video_frame_from_nv12(y, uv):
    h, w = y.shape
    frame = av.VideoFrame(w, h, "nv12")
    frame.planes[0].update(y.tobytes())
    frame.planes[1].update(uv.tobytes())
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--crf", default=None)
    parser.add_argument("--bitrate", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    encoder = args.encoder
    output_path = args.output or f"output_{encoder}.mp4"
    is_qsv = encoder.endswith("qsv")

    print(f"Encoder: {encoder}")
    print(f"Output: {output_path}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps, {args.frames} frames")

    print("Generating all frames in NumPy first...")
    t_gen = time.time()

    if is_qsv:
        numpy_frames = generate_nv12_frames(
            args.width, args.height, args.frames, seed=args.seed
        )
        pix_fmt = "nv12"
    else:
        numpy_frames = generate_yuv420p_frames(
            args.width, args.height, args.frames, seed=args.seed
        )
        pix_fmt = "yuv420p"

    print(f"Frame generation took {time.time() - t_gen:.2f}s")

    out_container = av.open(output_path, mode="w")
    out_video_stream = out_container.add_stream(encoder, rate=args.fps)
    out_video_stream.width = args.width
    out_video_stream.height = args.height
    out_video_stream.pix_fmt = pix_fmt

    if is_qsv:
        out_video_stream.options = {
            "preset": "6",
            # Try this too:
            # "async_depth": "8",
        }
    else:
        out_video_stream.options = {
            "preset": "6",
            "crf": args.crf or "30",
        }

    print("Starting encode...")
    t1 = time.time()

    for i, planes in enumerate(numpy_frames):
        if pix_fmt == "nv12":
            frame = video_frame_from_nv12(*planes)
        else:
            frame = video_frame_from_yuv420p(*planes)

        frame.pts = i

        for packet in out_video_stream.encode(frame):
            out_container.mux(packet)

        if i % args.fps == 0:
            print(f"  encoded {i}/{args.frames} frames")

    for packet in out_video_stream.encode():
        out_container.mux(packet)

    out_container.close()

    t2 = time.time() - t1
    print("Done.")
    print(f"Encode+wrap+mux time: {t2:.2f}s")
    print(f"Effective FPS: {args.frames / t2:.2f}")


if __name__ == "__main__":
    main()