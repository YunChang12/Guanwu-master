"""Dataset adapters — importing this package registers all built-in adapters."""

from guanwu.adapters import (  # noqa: F401  (side-effect: registers adapters)
    arkitscenes,
    maniskill3,
    objaverse_xl,
    partnet_mobility,
    procthor_10k,
    robotwin2,
    scannetpp,
)
