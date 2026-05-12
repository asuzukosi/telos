"""
contstant for the telos v1 format

pure python, no external dependencies
"""

# token mapping: telos v1
TELOS_TOKEN_MAP: tuple[tuple[str, int], ...] = (
    ("<|goal|>",     0),
    ("<|mission|>",  1),
    ("<|obs|>",      2),
    ("<|belief|>",   3),
    ("<|plan|>",     4),
    ("<|think|>",    5),
    ("<|action|>",   6),
    ("<|end|>",      7),
    ("<|result|>",   8),
    ("<|feedback|>", 9),
    ("<|reward|>",   10),
)
 
# ownership: which side of the loop is allowed to emit each marker.
TELOS_OWNERS: dict[str, str] = {
    "<|goal|>":     "runtime",
    "<|mission|>":  "runtime",
    "<|obs|>":      "runtime",
    "<|belief|>":   "model",
    "<|plan|>":     "model",
    "<|think|>":    "model",
    "<|action|>":   "model",
    "<|end|>":      "model",
    "<|result|>":   "runtime",
    "<|feedback|>": "runtime",
    "<|reward|>":   "runtime",
}
 
DEFAULT_BASE_MODEL: str = "meta-llama/Llama-3.1-8B"