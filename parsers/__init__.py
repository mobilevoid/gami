"""GAMI parser package — import all parsers to trigger registration."""
from parsers.base import (
    BaseParser,
    ParseResult,
    ParsedSegment,
    get_parser,
    get_parser_for_file,
    register_parser,
)
from parsers.markdown_parser import MarkdownParser
from parsers.conversation_parser import ConversationParser
from parsers.sqlite_memory_parser import SQLiteMemoryParser
from parsers.plaintext_parser import PlaintextParser

__all__ = [
    "BaseParser",
    "ParseResult",
    "ParsedSegment",
    "get_parser",
    "get_parser_for_file",
    "register_parser",
    "MarkdownParser",
    "ConversationParser",
    "SQLiteMemoryParser",
    "PlaintextParser",
]
