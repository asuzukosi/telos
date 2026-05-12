"""
frame definitions, paraser and render for th telos wire format
a talos trajectory is a sequence of frames. each frams opens with one of the 11 talos marker tokens and enxtends until the next marker token or end-of-string. there are no closing tokens
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from telos.constants import TELOS_OWNERS, TELOS_TOKEN_MAP