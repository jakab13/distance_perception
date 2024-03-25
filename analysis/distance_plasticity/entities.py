import slab
from dataclasses import dataclass


@dataclass
class Stimulus:
    id: int
    filename: str


@dataclass
class Response:
    value

