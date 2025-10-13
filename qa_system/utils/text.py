"""
Text utilities for QA system
===========================

Provides UTF-8 sanitization and text processing functions for robust
handling of generated content and reports.
"""

import re
import unicodedata
from typing import Optional


def sanitize_utf8(text: str) -> str:
    """
    Sanitize text for UTF-8 safe output.
    
    Args:
        text: Input text that may contain problematic characters
        
    Returns:
        Sanitized text safe for UTF-8 output
    """
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Replace problematic characters with safe alternatives
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201C': '"',  # Left double quotation mark
        '\u201D': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2022': '*',  # Bullet
        '\u2026': '...', # Horizontal ellipsis
        '\u00A0': ' ',  # Non-breaking space
        '\u00B0': ' degrees', # Degree sign
        '\u00B1': '+/-', # Plus-minus sign
        '\u00B2': '^2',  # Superscript two
        '\u00B3': '^3',  # Superscript three
        '\u00B5': 'u',   # Micro sign
        '\u00B9': '^1',  # Superscript one
        '\u00BA': ' degrees', # Masculine ordinal indicator
        '\u00BC': '1/4', # Vulgar fraction one quarter
        '\u00BD': '1/2', # Vulgar fraction one half
        '\u00BE': '3/4', # Vulgar fraction three quarters
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Ensure proper line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length while preserving word boundaries.
    
    Args:
        text: Input text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find the last space before max_length
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we found a space in the last 20%
        truncated = truncated[:last_space]
    
    return truncated + suffix


def extract_code_blocks(text: str) -> list[str]:
    """
    Extract code blocks from markdown-formatted text.
    
    Args:
        text: Text that may contain markdown code blocks
        
    Returns:
        List of code block contents
    """
    code_blocks = []
    
    # Match ```language\ncode\n``` patterns
    pattern = r'```(?:[a-zA-Z0-9_+-]*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        code_blocks.append(match.strip())
    
    return code_blocks


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count for text (rough approximation).
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Remove code blocks for more accurate estimation
    text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Count words and punctuation
    words = len(text_without_code.split())
    punctuation = len(re.findall(r'[^\w\s]', text_without_code))
    
    # Rough estimate: 1.3 tokens per word + punctuation
    return int(words * 1.3 + punctuation)


def clean_for_markdown(text: str) -> str:
    """
    Clean text for safe markdown output.
    
    Args:
        text: Text to clean
        
    Returns:
        Markdown-safe text
    """
    # First sanitize UTF-8
    text = sanitize_utf8(text)
    
    # Escape markdown special characters
    markdown_chars = ['*', '_', '`', '#', '+', '-', '.', '!', '[', ']', '(', ')', '|']
    for char in markdown_chars:
        text = text.replace(char, f'\\{char}')
    
    # Handle code blocks specially
    text = re.sub(r'```(.*?)```', r'`\1`', text, flags=re.DOTALL)
    
    return text


def extract_model_info(text: str) -> dict:
    """
    Extract model information from response text.
    
    Args:
        text: Response text that may contain model info
        
    Returns:
        Dictionary with extracted model information
    """
    info = {
        'model_name': 'unknown',
        'model_version': 'unknown',
        'generation_time': 'unknown'
    }
    
    # Look for common patterns
    model_patterns = [
        r'model[:\s]+([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)',
        r'generated by ([a-zA-Z0-9_-]+)',
        r'using ([a-zA-Z0-9_-]+) model'
    ]
    
    for pattern in model_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info['model_name'] = match.group(1)
            break
    
    return info


if __name__ == "__main__":
    # Test the utilities
    test_text = "Hello world! This is a test with some special chars: 'quotes', \"double quotes\", and em—dashes."
    print("Original:", test_text)
    print("Sanitized:", sanitize_utf8(test_text))
    print("Token estimate:", count_tokens_estimate(test_text))
