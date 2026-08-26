"""Port contract batteries.

Each module here holds **one** class of tests describing what a port promises. The batteries
are technology-free: they never import a concrete adapter. Per-adapter modules under ``tests/``
subclass a battery and supply only the fixture that constructs the adapter, so every adapter of
a port is held to the identical standard and substitutability is proven rather than asserted.
"""
