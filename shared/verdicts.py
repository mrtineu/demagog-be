"""Single source of truth for verdict mappings used across the entire project."""

# Slovak verdict → uppercase label
VERDICT_LABEL: dict[str, str] = {
    "Pravda": "PRAVDA",
    "Nepravda": "NEPRAVDA",
    "Zavádzajúce": "ZAVÁDZAJÚCE",
    "Neoveriteľné": "NEOVERITEĽNÉ",
}

# Slovak verdict → boolean correctness (None = uncheckable)
VERDICT_CORRECTNESS: dict[str, bool | None] = {
    "Pravda": True,
    "Nepravda": False,
    "Zavádzajúce": False,
    "Neoveriteľné": None,
}

# Slovak verdict → English key (used by statements/dashboard routes)
VERDICT_MAP: dict[str, str] = {
    "Pravda": "true",
    "Nepravda": "false",
    "Zavádzajúce": "misleading",
    "Neoveriteľné": "uncheckable",
    "Neviem posúdiť": "uncheckable",
}
