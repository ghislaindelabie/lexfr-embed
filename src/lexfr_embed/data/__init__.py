"""Data layer: load LegalKit, augment queries, mine hard negatives.

Pure, testable helpers (dedup / stratify / format) live alongside the network
loaders so the transformation logic can be unit-tested without downloads.
"""
