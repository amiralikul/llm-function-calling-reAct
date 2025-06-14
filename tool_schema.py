from __future__ import annotations

"""Centralised schema describing all function tools exposed to the LLM.

Importing this module *does not* pull in any heavy dependencies other than
`data_loader`, which is already cached by the interpreter once `main.py`
(or any other module) has loaded it.  Keeping the schema here makes
`main.py` shorter and easier to follow.
"""

import data_loader as dl

# Enums reused in the JSON schema definitions
category_enum = dl.category_enum
intent_enum = dl.intent_enum

# ---------------------------------------------------------------------------
# Tool specification list
# ---------------------------------------------------------------------------

tools: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "select_semantic_intent",
            "description": (
                "Filter the dataset by a list of intent names, cache the result, "
                "and return the cache key."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_names": {
                        "type": "array",
                        "items": {"type": "string", "enum": intent_enum},
                        "description": "List of intent names to filter by",
                    }
                },
                "required": ["intent_names"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_semantic_category",
            "description": (
                "Filter the dataset by a list of category names, cache the result, "
                "and return the cache key."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category_names": {
                        "type": "array",
                        "items": {"type": "string", "enum": category_enum},
                        "description": "List of category names to filter by",
                    }
                },
                "required": ["category_names"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sum",
            "description": "Function that sums two integers and returns the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first integer to sum"},
                    "b": {"type": "integer", "description": "The second integer to sum"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_intent",
            "description": "Count how many rows have the given intent name and return that number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_name": {
                        "type": "string",
                        "enum": intent_enum,
                        "description": "The intent name whose frequency you want to count.",
                    }
                },
                "required": ["intent_name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_category",
            "description": "Count how many rows have the given category name and return that number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_name": {
                        "type": "string",
                        "enum": category_enum,
                        "description": "The category name whose frequency you want to count.",
                    }
                },
                "required": ["category_name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_examples",
            "description": "Return a random sample of n examples from the cached dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The number of examples to show",
                    }
                },
                "required": ["n"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarise an arbitrary user request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The user request to summarise",
                    }
                },
                "required": ["user_request"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that the assistant now has enough data to answer the question and should produce a final response.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_intents",
            "description": "Return a list of all available intent names.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_categories",
            "description": "Return a list of all available category names.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        },
    },
]

__all__ = ["tools"] 