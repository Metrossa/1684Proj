"""
Text processing utilities for annotation storage optimization.
"""

import hashlib


def compute_text_hash(text: str) -> str:
    """
    Compute SHA256 hash of text for verification and deduplication.
    
    Args:
        text: Input text
        
    Returns:
        Hex string of SHA256 hash (first 16 chars for compactness)
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def get_text_preview(text: str, max_length: int = 100) -> str:
    """
    Get preview of text for display in hard cases.
    
    Args:
        text: Input text
        max_length: Maximum length of preview
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

