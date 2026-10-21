"""Compatibility re-export. NanoBench fetch lives in ``quadrotor.nanobench.fetch``."""

from quadrotor.nanobench.fetch import fetch, main

__all__ = ["fetch", "main"]

if __name__ == "__main__":
    main()
