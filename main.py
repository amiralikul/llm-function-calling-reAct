#import this dataset: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/viewer?views%5B%5D=train

import pandas as pd
import json
from typing import Literal

# Local helpers (dataset & OpenAI client)
import data_loader as dl
from llm_config import client

# Expose dataset-related globals used throughout this module
df = dl.df
category_enum = dl.category_enum
intent_enum = dl.intent_enum

# Shared cache starts as the full dataframe
CACHE = df

# After CACHE declaration, add global conversation storage
GLOBAL_MESSAGES: list = []  # Stores the full chat history across multiple `run()` invocations

# Centralised tool schema
from tool_schema import tools

def _cache_dataframe(df: pd.DataFrame) -> None:
    """Cache the provided DataFrame in the global CACHE variable.

    The *key* argument is now ignored and kept only for backward-compatibility
    so that existing calls that still pass a key do not break.  The refactoring
    changes the semantics of *CACHE* from a mapping of keys → DataFrames to a
    single slot that always contains **the most recently filtered DataFrame**.
    """
    global CACHE
    CACHE = df

def select_semantic_intent(intent_names: list[str]) -> str:
    """
    Filter the global dataframe by a list of intent names, cache the
    resulting subset, and return the cache key that should be used in
    subsequent calls (e.g. ``count_intent`` / ``count_category`` or ``show_examples``).

    Args:
        intent_names (list[str]): One or more intent names to filter by.

    Returns:
        str: The key under which the filtered dataframe is cached.
    """
    if isinstance(intent_names, str):  # allow accidental single-string usage
        intent_names = [intent_names]

    dataset = df[df["intent"].isin(intent_names)]
    _cache_dataframe(dataset)
    return f"Cached intents {intent_names} with {len(dataset)} rows"

def select_semantic_category(category_names: list[str]) -> str:
    """Same as ``select_semantic_intent`` but for categories."""
    if isinstance(category_names, str):
        category_names = [category_names]

    dataset = df[df["category"].isin(category_names)]
    _cache_dataframe(dataset)
    return f"Cached categories {category_names} with {len(dataset)} rows"
    

def get_all_intents() -> list[str]:
    return df["intent"].unique().tolist()

def get_all_categories() -> list[str]:
    return df["category"].unique().tolist()


def sum(a: int, b: int) -> int:
    return a + b

def show_examples(n: int) -> dict:
    if CACHE is None or len(CACHE) == 0:
        return {"error": "No data available"}
    if len(CACHE) < n:
        n = len(CACHE)
    return CACHE.sample(n).to_dict(orient="records")


def summarize(user_request: str) -> str:
    return f"Summary: {user_request}"

def finish() -> str:
    return "Conversation finished."
        

def plan(messages: list[dict], stream: bool = False) -> tuple[dict, str]:
    """Ask the LLM for an ordered list of tool calls *without* executing any tool.

    The model still receives the complete `tools` schema for reference, but we
    force `tool_choice='none'` so it cannot emit structured tool calls in the
    response. This lets the assistant know which tools exist while keeping the
    planning response purely textual.

    Parameters
    ----------
    messages : list[dict]
        The conversation history that should be provided when asking for a plan.
    stream : bool
        Whether to stream the response token-by-token.
    Returns
    -------
    tuple[dict, str]
        (1) The assistant message object returned by the OpenAI client so it can
            be appended to the global history, and (2) the plain-text content of
            that message for convenience/printing.
    """

    planning_messages = messages + [
        {
            "role": "system",
            "content": (
                "First, produce a concise ordered plan of the exact tool calls you will make "
                "(and their arguments) to satisfy the user's request. Do NOT execute any tool. "
                "Output the plan in plain text."
            ),
        }
    ]

    plan_completion_kwargs = {
        "model": "gpt-4o-mini",
        "messages": planning_messages,
        "tools": tools,         # Expose the schema so the model knows the signatures
        "tool_choice": "none", # Forbid tool execution during planning
    }

    if stream:
        # Stream token-by-token
        content_parts: list[str] = []
        for chunk in client.chat.completions.create(**plan_completion_kwargs, stream=True):
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)
        print()  # newline after stream is done
        full_content = "".join(content_parts)
        # Build a minimal assistant message dict
        plan_message = {"role": "assistant", "content": full_content}
        return plan_message, full_content
    else:
        plan_completion = client.chat.completions.create(**plan_completion_kwargs)
        plan_message = plan_completion.choices[0].message
        return plan_message, plan_message.content

# stream -> whether to stream assistant responses that are purely text (planning phase and final answer after finish)

def run(user_input: str, mode: str = 'react', stream: bool = False):
    # Accept both 'reAct' and 'react' (case-insensitive)
    mode = mode.lower()
    if mode not in {'react', 'planning'}:
        raise ValueError("mode must be either 'react/reAct' or 'planning'")

    global GLOBAL_MESSAGES

    # Initialise the conversation only the very first time this function is called
    if not GLOBAL_MESSAGES:
        GLOBAL_MESSAGES.append({
            "role": "system",
            "content": f"""You are a helpful assistant that can answer questions related to the customer support dataset.
        Each entry in the dataset contains the following fields:

        - instruction: a user request text
        - category: category of the user request
        - intent: the intent corresponding to the user instruction
        - response: an example expected response from the virtual assistant

        INSTRUCTIONS:

        1. Use the available tools to answer user questions.
        2. Some tools are dependent on other tools. You must use them in the correct order:
           - select_semantic_category() should be used before count_category()
           - select_semantic_intent() should be used before count_intent()

        3. IMPORTANT: Always call finish() when you have enough data to answer the question.
        4. You can make multiple tool calls in one response.
        5. If you have counted several categories/intents and can determine the biggest one, call finish() immediately.
        6. If the user’s question is not related to the dataset, respond **exactly** with:  
+          "this question is out of scope"  
+          (and do not call any tools).
        """
        })

    # Append the current user message to the global history
    GLOBAL_MESSAGES.append({"role": "user", "content": user_input})

    # Work with the shared history in the rest of the function
    messages = GLOBAL_MESSAGES
    
    # Planning mode: first ask the LLM for a plan, then ask it to execute that plan
    if mode == "planning":
        # Ask for the full plan first (no tool calls allowed)
        plan_message, plan_response = plan(messages, stream=stream)
        messages.append(plan_message)
        print("Planning step response:\n", plan_response)
        # Now ask the assistant to execute the plan
        messages.append({"role": "user", "content": "Please execute the plan step-by-step."})
        # Switch to react mode for execution stage
        mode = "react"

    while True:
        
        # Prepare parameters for the completion request depending on the chosen mode
        completion_kwargs = {
            "model": "gpt-4o-mini",
            "messages": messages,
        }
        if mode == "react":
            # In ReAct mode we expose the tool schema so the model can call them
            completion_kwargs["tools"] = tools
        # In planning mode we omit the 'tools' parameter entirely to discourage tool usage.

        completion = client.chat.completions.create(**completion_kwargs)

        choice = completion.choices[0]
        print(f"Response Content: {choice.message.content}")
        
        # Add the assistant's message to the conversation
        messages.append(choice.message)
        
        # Check if there are tool calls to execute
        if choice.message.tool_calls:
            print(f"\nTool Calls ({len(choice.message.tool_calls)}):")
            
            # Track if finish() was called
            finish_called = False
            
            for i, tool_call in enumerate(choice.message.tool_calls):
                tool_name = tool_call.function.name
                print(f"  Tool Call #{i+1}: {tool_name}")
                
                if tool_name == "finish":
                    finish_called = True
                
                try:
                    args = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool and get the result
                    if tool_name == "get_all_intents":
                        result = get_all_intents()
                    elif tool_name == "get_all_categories":
                        result = get_all_categories()
                    elif tool_name == "count_intent":
                        result = count_intent(args["intent_name"])
                    elif tool_name == "count_category":
                        result = count_category(args["category_name"])
                    elif tool_name == "sum":
                        result = sum(args["a"], args["b"])
                    elif tool_name == "show_examples":
                        result = show_examples(args["n"])
                    elif tool_name == "summarize":
                        result = summarize(args["user_request"])
                    elif tool_name == "select_semantic_intent":
                        result = select_semantic_intent(args["intent_names"])
                    elif tool_name == "select_semantic_category":
                        result = select_semantic_category(args["category_names"])

                    elif tool_name == "finish":
                        result = finish()
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    print(f"    Result: {result}")
                    
                    # Add the tool result to the conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                    
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    print(f"    Error: {error_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })
            
            # If finish() was called, get final response
            if finish_called:
                try:
                    final_completion_kwargs = {
                        "model": "gpt-4o-mini",
                        "messages": messages,
                    }
                    if stream:
                        full_content_parts: list[str] = []
                        for chunk in client.chat.completions.create(**final_completion_kwargs, stream=True):
                            delta = chunk.choices[0].delta
                            if delta.content:
                                print(delta.content, end="", flush=True)
                                full_content_parts.append(delta.content)
                        print()
                        final_response = "".join(full_content_parts)
                    else:
                        final_completion = client.chat.completions.create(**final_completion_kwargs)
                        final_response = final_completion.choices[0].message.content
                    print(f"\nFinal response: {final_response}")
                    return final_response
                except Exception as e:
                    error_msg = f"Error getting final response: {str(e)}"
                    print(error_msg)
                    return f"I encountered an error while processing your request: {error_msg}"
            
        else:
            # No tool calls, return the response
            return choice.message.content
    

# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------

def count_intent(intent_name: str) -> int:
    """Return the number of cached rows whose intent equals *intent_name*."""
    if CACHE is None or len(CACHE) == 0:
        return 0
    return int((CACHE["intent"] == intent_name).sum())


def count_category(category_name: str) -> int:
    """Return the number of cached rows whose category equals *category_name*."""
    if CACHE is None or len(CACHE) == 0:
        return 0
    return int((CACHE["category"] == category_name).sum())

if __name__ == "__main__":
        run("what is the most frequent intent", mode="react")

