# ExecuTorch Export Notes

- ExecuTorch support is optional and depends on extra packages and delegate availability.
- Treat ExecuTorch as edge/mobile oriented, not the default Runtime path.
- Verify operator coverage and delegate partitioning before presenting an artifact as deployable.
- Runtime core does not ship an ExecuTorch adapter in this package; use only when a documented companion adapter distribution is installed.
