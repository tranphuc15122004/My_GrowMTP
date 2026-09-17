"""Native recurrent MTP heads for the three models evaluated in GrowMTP."""

from .head import MTPHead, attach_mtp, shared_layers

__all__ = ["MTPHead", "attach_mtp", "shared_layers"]
