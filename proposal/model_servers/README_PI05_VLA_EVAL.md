# Pi0.5 with vla-eval (minimal)

Run from `/home/alfie/Develop/benchmarking-harness`.

## 1) Start model server

Terminal 1:

vla-eval serve --config model_servers/pi05_libero.yaml

## 2) Run LIBERO benchmark (Docker)

Terminal 2:

vla-eval run --config vla-evaluation-harness/configs/benchmarks/libero/smoke_test.yaml

For full LIBERO-10:

vla-eval run --config vla-evaluation-harness/configs/benchmarks/libero/10.yaml

## Notes

- Keep Terminal 1 running while Terminal 2 is evaluating.
- The benchmark config already targets `ws://localhost:8000`.
- If Docker permissions fail, ensure your user is in the `docker` group and open a new shell session.

If you previously ran a different server on port 8000, stop it first and restart with:

vla-eval serve --config model_servers/pi05_libero.yaml

## If you see `ModuleNotFoundError: No module named 'libero'`

This error is inside the benchmark Docker image (not your host env).

1) Re-pull the LIBERO image:

docker pull ghcr.io/allenai/vla-evaluation-harness/libero:latest

2) Verify the image can import `libero`:

docker run --rm --entrypoint python ghcr.io/allenai/vla-evaluation-harness/libero:latest \
	-c "import libero; print('libero OK')"

3) If import still fails, build the image locally from your checked-out harness:

cd vla-evaluation-harness
docker/build.sh --tag local libero

Then change `docker.image` in
`vla-evaluation-harness/configs/benchmarks/libero/smoke_test.yaml` to:

ghcr.io/allenai/vla-evaluation-harness/libero:local

and rerun `vla-eval run ...`.
