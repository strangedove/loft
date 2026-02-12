# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, deque
from collections.abc import Sequence
from itertools import takewhile
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, ProcessorMixin


DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)

# Role mappings for legacy conversation formats (human/gpt -> user/assistant)
ROLE_MAPPINGS = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}

# Default filler message for fix_turn_order
DEFAULT_TURN_ORDER_FILLER = "Let's begin."


def add_default_system_message(
    messages: list[dict[str, str]],
    system_message: str,
) -> list[dict[str, str]]:
    """
    Add a system message to a conversation if it doesn't already have one.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        system_message: The system message content to add.

    Returns:
        Messages with system message added at the beginning if not already present.

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ]
        >>> add_default_system_message(messages, "You are a helpful assistant.")
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    """
    if not messages:
        return messages

    # Check if first message is already a system message
    first_role = messages[0].get("role", "").lower()
    first_role = ROLE_MAPPINGS.get(first_role, first_role)

    if first_role == "system":
        # Already has a system message, don't modify
        return messages

    # Prepend system message
    return [{"role": "system", "content": system_message}] + messages


def add_system_message_to_example(
    example: dict[str, Any],
    system_message: str,
) -> dict[str, Any]:
    """
    Add a default system message to all conversation fields in an example.

    Checks for per-dataset `_system_message` column first, falling back to the provided
    global default.

    Args:
        example: A single data entry that may contain conversation fields.
        system_message: Default system message to add (global fallback).

    Returns:
        Example with system message added to all conversation fields.
    """
    result = dict(example)

    # Per-dataset system_message overrides global default
    effective_message = result.pop("_system_message", None) or system_message
    if not effective_message:
        return result

    # Add system message to all conversation-like keys
    for key in ["messages", "prompt", "chosen", "rejected", "completion", "conversations"]:
        if key in result and isinstance(result[key], list):
            messages = result[key]
            if messages and isinstance(messages[0], dict):
                # Check if it looks like a conversation (has role/content or from/value)
                first_msg = messages[0]
                if ("role" in first_msg and "content" in first_msg) or ("from" in first_msg and "value" in first_msg):
                    result[key] = add_default_system_message(messages, effective_message)

    return result


def fix_conversation_turn_order(
    messages: list[dict[str, str]],
    filler_message: str = DEFAULT_TURN_ORDER_FILLER,
) -> list[dict[str, str]]:
    """
    Fix conversation turn order to ensure strict system/user/assistant alternation.

    This function handles models with strict turn order requirements (e.g., Llama) by:
    1. Adding a filler user message if conversation starts with assistant (after optional system)
    2. Merging consecutive messages from the same role into one
    3. Dropping trailing user messages so conversations end with assistant

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        filler_message: Message to insert when a user turn is missing at the start.

    Returns:
        Fixed list of messages with proper turn order.

    Example:
        >>> messages = [
        ...     {"role": "assistant", "content": "Hello!"},
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "user", "content": "How are you?"},
        ...     {"role": "assistant", "content": "I'm good!"},
        ...     {"role": "user", "content": "Great"},
        ... ]
        >>> fix_conversation_turn_order(messages)
        [
            {"role": "user", "content": "Let's begin."},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Hi\n\nHow are you?"},
            {"role": "assistant", "content": "I'm good!"},
        ]
    """
    if not messages:
        return messages

    result = []

    # Step 1: Extract system message if present at the start
    system_message = None
    start_idx = 0
    if messages[0].get("role") == "system":
        system_message = messages[0]
        start_idx = 1

    # Check if we need to add a filler user message
    # (conversation starts with assistant after optional system)
    if start_idx < len(messages):
        first_non_system = messages[start_idx]
        first_role = first_non_system.get("role", "").lower()
        first_role = ROLE_MAPPINGS.get(first_role, first_role)

        if first_role == "assistant":
            # Add system message first if present
            if system_message:
                result.append(system_message)
            # Add filler user message
            result.append({"role": "user", "content": filler_message})
            # Continue from assistant message
        elif system_message:
            result.append(system_message)
    elif system_message:
        result.append(system_message)

    # Step 2: Process remaining messages, merging consecutive same-role messages
    for msg in messages[start_idx:]:
        role = msg.get("role", "").lower()
        role = ROLE_MAPPINGS.get(role, role)
        content = msg.get("content", "")

        if not result:
            result.append({"role": role, "content": content})
        elif result[-1]["role"] == role:
            # Merge with previous message (same role)
            result[-1]["content"] = result[-1]["content"] + "\n\n" + content
        else:
            result.append({"role": role, "content": content})

    # Step 3: Drop trailing user messages (conversation should end with assistant)
    while result and result[-1].get("role") == "user":
        result.pop()

    # If we removed everything except system, return empty
    if len(result) <= 1 and result and result[0].get("role") == "system":
        return []

    return result


def fix_example_turn_order(
    example: dict[str, Any],
    filler_message: str = DEFAULT_TURN_ORDER_FILLER,
) -> dict[str, Any]:
    """
    Apply fix_conversation_turn_order to all conversation fields in an example.

    Args:
        example: A single data entry that may contain conversation fields.
        filler_message: Message to insert when a user turn is missing.

    Returns:
        Example with fixed turn order in all conversation fields.
    """
    result = dict(example)

    # Fix turn order for all conversation-like keys
    for key in ["messages", "prompt", "chosen", "rejected", "completion", "conversations"]:
        if key in result and isinstance(result[key], list):
            messages = result[key]
            if messages and isinstance(messages[0], dict):
                # Check if it looks like a conversation (has role/content or from/value)
                first_msg = messages[0]
                if ("role" in first_msg and "content" in first_msg) or ("from" in first_msg and "value" in first_msg):
                    result[key] = fix_conversation_turn_order(messages, filler_message)

    return result


def prepare_multimodal_messages(messages: list[dict[str, Any]], num_images: int) -> None:
    """
    Convert messages into a structured multimodal format if needed.

    Each message's content is transformed from a raw string into a list of typed parts. The first user message is
    prefixed with an image placeholder, while all other user and assistant messages are wrapped as text entries.

    Args:
        messages (`list[dict[str, Any]]`):
            Messages with `"role"` and `"content"`. Content may be a raw string before transformation.
        num_images (`int`):
            Number of images to include in the first user message. This is used to determine how many image
            placeholders to add.

    Example:
    ```python
    # Input
    [
        {"role": "user", "content": "What's in this image?"},
        {"role": "assistant", "content": "It looks like a cat."},
    ]

    # Output (num_images=1)
    [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What's in this image?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It looks like a cat."}]},
    ]
    ```
    """
    image_included = False
    for message in messages:
        if message["role"] == "system":
            if isinstance(message["content"], str):  # if already prepared, the content will be a list
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "user":
            if isinstance(message["content"], str) and not image_included:
                placeholders = [{"type": "image"}] * num_images
                message["content"] = [*placeholders, {"type": "text", "text": message["content"]}]
                image_included = True
            elif isinstance(message["content"], str) and image_included:
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "assistant":
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        else:
            raise ValueError(f"Invalid role in message: {message['role']}. Expected 'user', 'assistant', or 'system'.")


def is_conversational(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format.

    Supports both standard format (role/content) and legacy format (from/value with conversations key).

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True

    >>> example = {"prompt": "The sky is"}
    >>> is_conversational(example)
    False

    >>> example = {"conversations": [{"from": "human", "value": "Hello"}]}
    >>> is_conversational(example)
    True
    ```
    """
    # Support both standard keys and legacy 'conversations' key
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "conversations"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages
        if isinstance(maybe_messages, list) and len(maybe_messages) > 0:
            maybe_message = maybe_messages[0]
            if isinstance(maybe_message, dict):
                # Standard format: "role" and "content"
                if "role" in maybe_message and "content" in maybe_message:
                    return True
                # Legacy format: "from" and "value"
                if "from" in maybe_message and "value" in maybe_message:
                    return True

    return False


def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: Union[PreTrainedTokenizerBase, ProcessorMixin],
    tools: Optional[list[Union[dict, Callable]]] = None,
    **template_kwargs,
) -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # language modeling
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False, **template_kwargs)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"], tools=tools, tokenize=False, **template_kwargs
            )
            # DeepSeek-R1 inserts a <tool_call> token when using `add_generation_prompt`, which can cause discrepancies
            # between the prompt alone and the combined prompt+completion. To ensure consistency, we extract the
            # common prefix between the two. In most cases, this is a no-op.
            prompt = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt, prompt_chosen)))

            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"], tools=tools, tokenize=False, **template_kwargs
            )
            # Handle DeepSeek-R1 <tool_call> token, see the above comment for details
            prompt = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt, prompt_rejected)))
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tools=tools, tokenize=False, **template_kwargs
            )
            # Handle DeepSeek-R1 <tool_call> token, see the above comment for details
            prompt = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt, prompt_completion)))
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False, **template_kwargs)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(
                example["rejected"], tools=tools, tokenize=False, **template_kwargs
            )

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "chosen" in example:
        output["chosen"] = chosen
    if "rejected" in example:
        output["rejected"] = rejected
    if "completion" in example:
        output["completion"] = completion
    if "label" in example:
        output["label"] = example["label"]

    return output


def maybe_apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[Union[dict, Callable]]] = None,
    **template_kwargs: Any,
) -> dict[str, str]:
    r"""
    If the example is in a conversational format, apply a chat template to it.

    Args:
        example (`dict[str, list[dict[str, str]]`):
            Dictionary representing a single data entry of a conversational dataset. Each data entry can have different
            keys depending on the dataset type. The supported dataset types are:

                - Language modeling dataset: `"messages"`.
                - Prompt-only dataset: `"prompt"`.
                - Prompt-completion dataset: `"prompt"` and `"completion"`.
                - Preference dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Preference dataset with implicit prompt: `"chosen"` and `"rejected"`.
                - Unpaired preference dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"messages"`, `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of
            messages, where each message is a dictionary with keys `"role"` and `"content"`.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer to apply the chat template with.
        tools (`list[Union[dict, Callable]]`, *optional*):
            A list of tools (callable functions) that will be accessible to the model. If the template does not support
            function calling, this argument will have no effect.
        **template_kwargs (`Any`, *optional*):
            Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

    Returns:
        `dict[str, str]`:
            Formatted example with the chat template applied.

    Notes:
        - This function does not alter the keys, except for Language modeling dataset, where `"messages"` is replaced
        by `"text"`.

        - In case of prompt-only data, if the last role is `"user"`, the generation prompt is added to the prompt.
        Else, if the last role is `"assistant"`, the final message is continued.

    Example:

    ```python
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    >>> example = {
    ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
    ...     "completion": [{"role": "assistant", "content": "It is blue."}],
    ... }
    >>> apply_chat_template(example, tokenizer)
    {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n'}
    ```
    """
    if is_conversational(example):
        # Normalize legacy formats (conversations -> messages, from/value -> role/content, human/gpt -> user/assistant)
        example = maybe_convert_to_chatml(example)
        return apply_chat_template(example, tokenizer, tools, **template_kwargs)
    else:
        return example


def _unpair_row(examples: list[dict[str, list[dict[str, str]]]]) -> list[dict[str, list[dict[str, str]]]]:
    batch_size = len(examples["chosen"])
    new_rows = {
        "completion": examples["chosen"] + examples["rejected"],
        "label": [True] * batch_size + [False] * batch_size,
    }
    # Duplicate all columns except chosen/rejected to match the new row count
    for key in examples:
        if key not in ("chosen", "rejected"):
            new_rows[key] = examples[key] + examples[key]
    return new_rows


def unpair_preference_dataset(
    dataset: DatasetType, num_proc: Optional[int] = None, desc: Optional[str] = None
) -> DatasetType:
    r"""
    Unpair a preference dataset.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        desc (`str`, *optional*):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        `Dataset`: The unpaired preference dataset.

    Example:

    ```python
    >>> from datasets import Dataset

    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"],
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })

    >>> dataset[0]
    {'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
    ```
    """
    return dataset.map(_unpair_row, batched=True, remove_columns=["chosen", "rejected"], num_proc=num_proc, desc=desc)


def maybe_unpair_preference_dataset(
    dataset: DatasetType, num_proc: Optional[int] = None, desc: Optional[str] = None
) -> DatasetType:
    r"""
    Unpair a preference dataset if it is paired.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        desc (`str`, *optional*):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        `Dataset` or `DatasetDict`: The unpaired preference dataset if it was paired, otherwise the original dataset.

    Example:

    ```python
    >>> from datasets import Dataset

    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"],
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })

    >>> dataset[0]
    {'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
    ```
    """
    if isinstance(dataset, DatasetDict):
        column_names = dataset[list(dataset.keys())[0]].column_names
    else:
        column_names = dataset.column_names
    if "chosen" in column_names and "rejected" in column_names:
        return unpair_preference_dataset(dataset, num_proc=num_proc, desc=desc)
    else:
        return dataset


def extract_prompt(example: dict[str, Sequence]) -> dict[str, Sequence]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both the chosen and
    rejected completions.

    For more details, see [`maybe_extract_prompt`].
    """
    for idx in range(min(len(example["chosen"]), len(example["rejected"]))):
        if example["chosen"][idx] != example["rejected"][idx]:
            if example["chosen"][idx - 1] == " ":  # remove space before the prompt
                idx -= 1
            break
    return {
        "prompt": example["chosen"][:idx],
        "chosen": example["chosen"][idx:],
        "rejected": example["rejected"][idx:],
    }


def maybe_extract_prompt(example: dict[str, list]) -> dict[str, list]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both the chosen and
    rejected completions.

    If the example already contains a `"prompt"` key, the function returns the example as is. Else, the function
    identifies the longest common sequence (prefix) of conversation turns between the "chosen" and "rejected"
    completions and extracts this as the prompt. It then removes this prompt from the respective "chosen" and
    "rejected" completions.

    Args:
        example (`dict[str, list]`):
            A dictionary representing a single data entry in the preference dataset. It must contain the keys
            `"chosen"` and `"rejected"`, where each value is either conversational or standard (`str`).

    Returns:
        `dict[str, list]`: A dictionary containing:
            - `"prompt"`: The longest common prefix between the "chosen" and "rejected" completions.
            - `"chosen"`: The remainder of the "chosen" completion, with the prompt removed.
            - `"rejected"`: The remainder of the "rejected" completion, with the prompt removed.

    Examples:

    ```python
    >>> example = {
    ...     "chosen": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is blue."},
    ...     ],
    ...     "rejected": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is green."},
    ...     ],
    ... }
    >>> extract_prompt(example)
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```

    Or, with the `map` method of `datasets.Dataset`:

    ```python
    >>> from trl import extract_prompt
    >>> from datasets import Dataset

    >>> dataset_dict = {
    ...     "chosen": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is blue."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sky."},
    ...         ],
    ...     ],
    ...     "rejected": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is green."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sea."},
    ...         ],
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = dataset.map(extract_prompt)
    >>> dataset[0]
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```
    """
    # Some dataset add a `"prompt"` column, even though the prompt is implicit and included in the "chosen" and
    # "rejected" completions. E.g.:
    # {"prompt": "What color is the sky?",
    #  "chosen": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
    #  "rejected": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}]}
    # That's why we check if the prompt is also conversational before deciding not to extract it.
    if "chosen" not in example or "rejected" not in example:  # not a preference example
        return example
    if "prompt" in example:
        # Both conversational or both non-conversational
        chosen_conv = is_conversational({"chosen": example["chosen"]})
        prompt_conv = is_conversational({"prompt": example["prompt"]})
        if (chosen_conv and prompt_conv) or (not chosen_conv and not prompt_conv):
            return example
    return extract_prompt({"chosen": example["chosen"], "rejected": example["rejected"]})


class _SegmentTree:
    """
    A segment tree data structure that, when initialized as `_SegmentTree(maxval)`, efficiently finds the next larger
    value for a given input within the range [1, maxval].

    See [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) for more details.
    """

    def __init__(self, maxval: int):
        self.maxval = maxval
        # For non-power-of-2 values, we need to round up to the next power of 2 for the tree size
        self.tree_size = 1 << (maxval - 1).bit_length()
        self.tree = [0] * (2 * self.tree_size)

    def add(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = val
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            # Compare the values using if-else otherwise repeated calls to `builtins.max` become the bottleneck
            self.tree[i] = left if left >= right else right

    def remove(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = 0
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            # Compare the values using if-else otherwise repeated calls to `builtins.max` become the bottleneck
            self.tree[i] = left if left >= right else right

    def search(self, val):
        assert 0 < val <= self.maxval
        i = 1
        while i < self.tree_size:
            if self.tree[i << 1] >= val:
                i = i << 1
            else:
                i = (i << 1) + 1
        return self.tree[i]


def _pack_bfd(examples: pa.Table, seq_length: int) -> pa.Table:
    """Pack sequences in a pyarrow Table using Best Fit Decreasing strategy."""
    columns = []
    list_column_idx = None
    for idx, column in enumerate(examples.columns):
        if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
            column = pc.list_slice(column, 0, seq_length)
            if list_column_idx is None:
                list_column_idx = idx
        columns.append(column)
    examples = pa.Table.from_arrays(columns, names=examples.column_names)

    ids = np.arange(len(examples))
    assert list_column_idx is not None
    lengths = pc.list_value_length(examples[list_column_idx]).combine_chunks()
    examples = examples.append_column("seq_lengths", lengths)  # Allows us to later construct `position_ids`
    lengths = pc.make_struct(lengths, ids)
    lengths = lengths.sort("descending", by=0)

    segment_tree = _SegmentTree(seq_length)
    segment_tree.add(seq_length)  # the max, `seq_length` bin is always available
    space_to_bin = defaultdict(deque)

    # Bin is represented as a dict (of example ids and sum of their lengths) to allow in-place updates
    bins: list[dict] = []
    for length, idx in zip(lengths.field(0).to_numpy(), lengths.field(1).to_numpy()):
        space = segment_tree.search(length)

        if space < seq_length:
            # Use existing bin with exactly this amount of space
            bin = space_to_bin[space].popleft()
        else:
            # Create a new bin
            bin = {"ids": [], "length": 0}
            bins.append(bin)

        bin["ids"].append(idx)
        bin["length"] += length
        if space < seq_length and not space_to_bin[space]:
            segment_tree.remove(space)

        space = space - length
        space_to_bin[space].append(bin)
        if space > 0:
            segment_tree.add(space)

    examples = pc.take(examples, [id_ for bin in bins for id_ in bin["ids"]])
    offsets = np.array([0] + [bin["length"] for bin in bins])
    offsets = np.cumsum(offsets)

    assert all(
        column.num_chunks == 1 for column in examples.columns
    )  # `pc.take` returns a ChunkedArray with a single chunk

    lengths = examples["seq_lengths"].chunks[0]
    examples = examples.drop_columns("seq_lengths")
    lengths = pa.ListArray.from_arrays(np.cumsum([0] + [len(bin["ids"]) for bin in bins], dtype=np.int32), lengths)

    columns = []
    for column in examples.columns:
        column = column.chunks[0]
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            dtype = column.offsets.type.to_pandas_dtype()
            column = type(column).from_arrays(offsets.astype(dtype), column.values)
        columns.append(column)
    return pa.Table.from_arrays(columns + [lengths], names=examples.column_names + ["seq_lengths"])


def _pack_wrapped(examples: pa.Table, seq_length: int) -> pa.Table:
    """Pack sequences in a pyarrow Table using a wrapped strategy."""
    columns = []
    for column in examples.columns:
        if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
            if isinstance(column, pa.ChunkedArray):
                column = column.combine_chunks()
            offsets, values = column.offsets, column.values
            values = values[offsets[0].as_py() : offsets[-1].as_py()]
            num_elements = len(values)
            dtype = offsets.type.to_pandas_dtype()  # np.int32 or np.int64
            offsets = np.arange(0, num_elements, seq_length, dtype=dtype)
            offsets = np.concatenate((offsets, [num_elements]))
            column = type(column).from_arrays(offsets, values)
        columns.append(column)
    return pa.Table.from_arrays(columns, names=examples.column_names)


def pack_dataset(
    dataset: DatasetType, seq_length: int, strategy: str = "bfd", map_kwargs: Optional[dict[str, Any]] = None
) -> DatasetType:
    r"""
    Pack sequences in a dataset into chunks of size `seq_length`.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Dataset to pack
        seq_length (`int`):
            Target sequence length to pack to.
        strategy (`str`, *optional*, defaults to `"bfd"`):
            Packing strategy to use. Can be either:

            - `"bfd"` (Best Fit Decreasing): Slower but preserves sequence boundaries. Sequences are never cut in the
                middle.
            - `"wrapped"`: Faster but more aggressive. Ignores sequence boundaries and will cut sequences in the middle
                to completely fill each packed sequence with data.
        map_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the dataset's map method when packing examples.

    Returns:
        `Dataset` or `DatasetDict`: The dataset with packed sequences. The number of examples may decrease as sequences
        are combined.

    Example:
    ```python
    >>> from datasets import Dataset
    >>> from trl import pack_dataset

    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5], [6, 7, 8], [9]],
    ...     "attention_mask": [[1, 1, 0], [1, 0], [1, 0, 0], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> packed_dataset = pack_dataset(dataset, seq_length=4, strategy="bfd")
    >>> packed_dataset[:]
    {'input_ids': [[1, 2, 3, 9], [6, 7, 8], [4, 5]],
    'attention_mask': [[1, 1, 0, 1], [1, 0, 0], [1, 0]],
    'seq_lengths': [[3, 1], [3], [2]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}
    # Fast packing with pyarrow
    dataset = dataset.with_format("arrow")
    if strategy == "bfd":
        dataset = dataset.map(_pack_bfd, batched=True, fn_kwargs={"seq_length": seq_length}, **map_kwargs)
    elif strategy == "wrapped":
        dataset = dataset.map(_pack_wrapped, batched=True, fn_kwargs={"seq_length": seq_length}, **map_kwargs)
    else:
        raise ValueError(f"Invalid packing strategy: {strategy}. Use 'bfd' or 'wrapped'.")
    dataset = dataset.with_format(None)
    return dataset


def truncate_dataset(
    dataset: DatasetType, max_length: int, map_kwargs: Optional[dict[str, Any]] = None
) -> DatasetType:
    r"""
    Truncate sequences in a dataset to a specified `max_length`.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Dataset to truncate.
        max_length (`int`):
            Maximum sequence length to truncate to.
        map_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the dataset's map method when truncating examples.

    Returns:
        `Dataset` or `DatasetDict`: The dataset with truncated sequences.

    Example:
    ```python
    >>> from datasets import Dataset

    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> truncated_dataset = truncate_dataset(dataset, max_length=2)
    >>> truncated_dataset[:]
    {'input_ids': [[1, 2], [4, 5], [8]],
     'attention_mask': [[0, 1], [0, 0], [1]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}
    if isinstance(dataset, Dataset):
        # Fast truncation with pyarrow
        def truncate(examples):
            truncated_columns = []
            for column in examples.columns:
                if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
                    column = pc.list_slice(column, 0, max_length)
                truncated_columns.append(column)
            return pa.Table.from_arrays(truncated_columns, names=examples.column_names)

        dataset = dataset.with_format("arrow")
        dataset = dataset.map(truncate, batched=True, **map_kwargs)
        dataset = dataset.with_format(None)
    else:

        def truncate(examples):
            truncated_examples = {}
            for key, column in examples.items():
                if column and isinstance(column[0], list):
                    column = [val[:max_length] for val in column]
                truncated_examples[key] = column
            return truncated_examples

        dataset = dataset.map(
            truncate,
            batched=True,
            **map_kwargs,
        )
    return dataset


def is_conversational_from_value(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format (from/value). Note that this format isn't recommended. Prefer
    the ChatML format (role/content)

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational Chatformat, `False` otherwise.

    Examples:

    ```python
    >>> example = {"conversations": [{"from": "user", "value": "What color is the sky?"}]}
    >>> is_conversational_from_value(example)
    True

    >>> example = {"conversations": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational_from_value(example)
    False

    >>> example = {"conversations": "The sky is"}
    >>> is_conversational_from_value(example)
    False
    ```
    """
    maybe_messages = example.get("conversations")
    # It must be a list of messages
    if isinstance(maybe_messages, list):
        maybe_message = maybe_messages[0]
        # Each message must a list of dictionaries with keys "from" and "value"
        if isinstance(maybe_message, dict) and "from" in maybe_message and "value" in maybe_message:
            return True

    return False


def maybe_convert_to_chatml(example: dict[str, list]) -> dict[str, list]:
    """
    Convert a conversational dataset with fields `from` and `value` to ChatML format.

    This function modifies conversational data to align with OpenAI's ChatML format:
    - Replaces the key `"from"` with `"role"` in message dictionaries.
    - Replaces the key `"value"` with `"content"` in message dictionaries.
    - Renames `"conversations"` to `"messages"` for consistency with ChatML.
    - Maps legacy role names: `"human"` → `"user"`, `"gpt"` → `"assistant"`.

    Args:
        example (`dict[str, list]`):
            A single data entry containing a list of messages.

    Returns:
        `dict[str, list]`:
            Example reformatted to ChatML style.

    Example:
    ```python
    >>> from trl import maybe_convert_to_chatml

    >>> example = {
    ...     "conversations": [
    ...         {"from": "human", "value": "What color is the sky?"},
    ...         {"from": "gpt", "value": "It is blue."},
    ...     ]
    ... }
    >>> maybe_convert_to_chatml(example)
    {'messages': [{'role': 'user', 'content': 'What color is the sky?'},
                  {'role': 'assistant', 'content': 'It is blue.'}]}
    ```
    """
    # List of possible keys containing message lists
    for key in ["prompt", "completion", "chosen", "rejected", "messages", "conversations"]:
        if key in example and isinstance(example[key], list):
            messages = example[key]
            for message in messages:
                if isinstance(message, dict):
                    if "from" in message:
                        role = message.pop("from")
                        # Map legacy role names to standard ones
                        role = ROLE_MAPPINGS.get(role.lower(), role)
                        message["role"] = role
                    elif "role" in message:
                        # Also normalize role names in standard format
                        role = message["role"]
                        message["role"] = ROLE_MAPPINGS.get(role.lower(), role)
                    if "value" in message:
                        message["content"] = message.pop("value")

    # Rename "conversations" to "messages"
    if "conversations" in example:
        example["messages"] = example.pop("conversations")

    return example


def is_preference_dataset(example: dict[str, Any]) -> bool:
    """
    Check if the example is from a preference dataset (has chosen/rejected columns).

    Args:
        example: A single data entry of a dataset.

    Returns:
        True if the example has 'chosen' and 'rejected' keys.
    """
    return "chosen" in example and "rejected" in example


def is_binary_preference_dataset(example: dict[str, Any]) -> bool:
    """
    Check if the example is from a binary preference dataset (has completion and label).

    This is the KTO-style format where each example has a completion and a boolean label
    indicating whether it's a good (True) or bad (False) response.

    Args:
        example: A single data entry of a dataset.

    Returns:
        True if the example has 'completion' and 'label' keys.
    """
    return "completion" in example and "label" in example


def convert_preference_to_sft(example: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a preference dataset example to SFT format.

    Takes a preference example with prompt/chosen/rejected and converts it to
    a conversational SFT format using only the chosen response:
    - prompt -> user message(s)
    - chosen -> assistant message(s)

    Handles both conversational and string formats.

    Args:
        example: A preference dataset example with 'prompt', 'chosen', 'rejected' keys.

    Returns:
        An SFT example with 'messages' key containing the conversation.

    Example:
        >>> example = {
        ...     "prompt": [{"role": "user", "content": "Hello"}],
        ...     "chosen": [{"role": "assistant", "content": "Hi there!"}],
        ...     "rejected": [{"role": "assistant", "content": "Go away"}],
        ... }
        >>> convert_preference_to_sft(example)
        {'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}]}

        >>> example = {
        ...     "prompt": "What is 2+2?",
        ...     "chosen": "4",
        ...     "rejected": "5",
        ... }
        >>> convert_preference_to_sft(example)
        {'messages': [{'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant', 'content': '4'}]}
    """
    result = {}

    # Copy over any extra keys (except the preference-specific ones)
    for key in example:
        if key not in ("prompt", "chosen", "rejected"):
            result[key] = example[key]

    prompt = example.get("prompt", [])
    chosen = example.get("chosen", [])

    # Handle conversational format
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        messages = list(prompt)  # Copy prompt messages
        if isinstance(chosen, list) and chosen and isinstance(chosen[0], dict):
            messages.extend(chosen)
        elif isinstance(chosen, str):
            messages.append({"role": "assistant", "content": chosen})
        result["messages"] = messages

    # Handle string format
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
        if isinstance(chosen, str):
            messages.append({"role": "assistant", "content": chosen})
        elif isinstance(chosen, list) and chosen and isinstance(chosen[0], dict):
            messages.extend(chosen)
        result["messages"] = messages

    # Handle implicit prompt (chosen/rejected only, no prompt)
    elif not prompt and isinstance(chosen, list) and chosen and isinstance(chosen[0], dict):
        # The chosen already contains the full conversation
        result["messages"] = list(chosen)

    return result


def convert_binary_preference_to_sft(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Convert a binary preference (KTO-style) example to SFT format.

    Takes a binary preference example with prompt/completion/label and converts
    good examples (label=True) to conversational SFT format. Bad examples are
    filtered out by returning None.

    Args:
        example: A binary preference example with 'prompt', 'completion', 'label' keys.

    Returns:
        An SFT example with 'messages' key if label is True, None otherwise.

    Example:
        >>> example = {
        ...     "prompt": [{"role": "user", "content": "Hello"}],
        ...     "completion": [{"role": "assistant", "content": "Hi!"}],
        ...     "label": True,
        ... }
        >>> convert_binary_preference_to_sft(example)
        {'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi!'}]}

        >>> example = {"prompt": "Hi", "completion": "Bad response", "label": False}
        >>> convert_binary_preference_to_sft(example)  # Returns None (filtered out)
    """
    # Only keep good examples
    if not example.get("label", False):
        return None

    result = {}

    # Copy over any extra keys (except the preference-specific ones)
    for key in example:
        if key not in ("prompt", "completion", "label"):
            result[key] = example[key]

    prompt = example.get("prompt", [])
    completion = example.get("completion", [])

    # Handle conversational format
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        messages = list(prompt)  # Copy prompt messages
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            messages.extend(completion)
        elif isinstance(completion, str):
            messages.append({"role": "assistant", "content": completion})
        result["messages"] = messages

    # Handle string format
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
        if isinstance(completion, str):
            messages.append({"role": "assistant", "content": completion})
        elif isinstance(completion, list) and completion and isinstance(completion[0], dict):
            messages.extend(completion)
        result["messages"] = messages

    return result


# =============================================================================
# Truncation Strategies
# =============================================================================

VALID_TRUNCATION_STRATEGIES = {"truncate", "drop", "split", "truncate_turns"}


def truncate_tokens_with_strategy(
    input_ids: list[int],
    attention_mask: list[int],
    max_length: int,
    eos_token_id: int,
    bos_token_id: Optional[int] = None,
    strategy: str = "truncate",
) -> Optional[tuple[list[int], list[int]]]:
    """
    Truncate tokenized sequences according to the specified strategy.

    Args:
        input_ids: Token IDs to truncate.
        attention_mask: Attention mask to truncate.
        max_length: Maximum sequence length.
        eos_token_id: EOS token ID (removed from truncated sequences).
        bos_token_id: BOS token ID (optional, for split strategy).
        strategy: Truncation strategy ("truncate" or "drop").

    Returns:
        Tuple of (input_ids, attention_mask), or None if sample should be dropped.

    Note:
        For "split" strategy, use `split_tokens_into_chunks` instead.
        For "truncate_turns", use `truncate_conversation_by_turns` instead.
    """
    if len(input_ids) <= max_length:
        return input_ids, attention_mask

    if strategy == "drop":
        return None

    if strategy == "truncate":
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        # Remove trailing EOS if present (truncated sequences shouldn't end with EOS)
        if input_ids and input_ids[-1] == eos_token_id:
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
        return input_ids, attention_mask

    raise ValueError(f"Invalid truncation strategy: {strategy}. Use 'truncate' or 'drop'.")


def split_tokens_into_chunks(
    input_ids: list[int],
    attention_mask: list[int],
    max_length: int,
    eos_token_id: int,
    bos_token_id: Optional[int] = None,
) -> list[tuple[list[int], list[int]]]:
    """
    Split tokenized sequence into multiple chunks for continued pretraining.

    - First chunk gets BOS (if present in original)
    - Last chunk gets EOS
    - Middle chunks get neither

    Args:
        input_ids: Token IDs to split.
        attention_mask: Attention mask to split.
        max_length: Maximum length per chunk.
        eos_token_id: EOS token ID.
        bos_token_id: BOS token ID (optional).

    Returns:
        List of (input_ids, attention_mask) tuples for each chunk.
    """
    if len(input_ids) <= max_length:
        return [(input_ids, attention_mask)]

    # Check if original starts with BOS
    has_bos = bos_token_id is not None and input_ids and input_ids[0] == bos_token_id
    # Check if original ends with EOS
    has_eos = input_ids and input_ids[-1] == eos_token_id

    # Strip BOS/EOS for chunking
    content_ids = input_ids
    content_mask = attention_mask
    if has_bos:
        content_ids = content_ids[1:]
        content_mask = content_mask[1:]
    if has_eos:
        content_ids = content_ids[:-1]
        content_mask = content_mask[:-1]

    chunks = []
    # Calculate effective chunk size (accounting for BOS/EOS we'll add)
    first_chunk_size = max_length - (1 if has_bos else 0)
    last_chunk_needs_eos = has_eos
    middle_chunk_size = max_length

    pos = 0
    chunk_idx = 0
    while pos < len(content_ids):
        is_first = chunk_idx == 0
        remaining = len(content_ids) - pos
        is_last = remaining <= (max_length - (1 if last_chunk_needs_eos else 0))

        if is_first and has_bos:
            chunk_size = first_chunk_size
        elif is_last and last_chunk_needs_eos:
            chunk_size = max_length - 1
        else:
            chunk_size = middle_chunk_size

        chunk_ids = content_ids[pos : pos + chunk_size]
        chunk_mask = content_mask[pos : pos + chunk_size]

        # Add BOS to first chunk
        if is_first and has_bos:
            chunk_ids = [bos_token_id] + chunk_ids
            chunk_mask = [1] + chunk_mask

        # Add EOS to last chunk
        if is_last and last_chunk_needs_eos:
            chunk_ids = chunk_ids + [eos_token_id]
            chunk_mask = chunk_mask + [1]

        chunks.append((chunk_ids, chunk_mask))
        pos += chunk_size
        chunk_idx += 1

    return chunks


def truncate_conversation_by_turns(
    messages: list[dict[str, str]],
    tokenizer,
    max_length: int,
    chat_template: Optional[str] = None,
) -> Optional[list[dict[str, str]]]:
    """
    Truncate a conversation by dropping complete turn pairs from the end.

    Preserves the system message (if present) and keeps turn pairs from the
    beginning until the conversation fits within max_length.
    If even a single turn pair (plus system) exceeds max_length, returns None.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        tokenizer: Tokenizer to use for length calculation.
        max_length: Maximum token length.
        chat_template: Optional chat template to use.

    Returns:
        Truncated messages list, or None if the sample should be dropped.
    """
    if not messages:
        return None

    # Separate system message if present
    system_msg = None
    conversation = messages
    if messages[0].get("role") == "system":
        system_msg = messages[0]
        conversation = messages[1:]

    if not conversation:
        return None

    # Group into turn pairs (user + assistant)
    # Handle edge cases where conversation might not be perfectly paired
    turn_pairs = []
    i = 0
    while i < len(conversation):
        pair = [conversation[i]]
        i += 1
        # Collect any following messages until we hit another user message
        while i < len(conversation) and conversation[i].get("role") != "user":
            pair.append(conversation[i])
            i += 1
        turn_pairs.append(pair)

    # Calculate token length for system message
    system_tokens = 0
    if system_msg:
        system_text = tokenizer.apply_chat_template(
            [system_msg], tokenize=True, add_generation_prompt=False, chat_template=chat_template
        )
        system_tokens = len(system_text)

    # Calculate token lengths for each turn pair
    pair_tokens = []
    for pair in turn_pairs:
        # Tokenize the pair in context
        if system_msg:
            full_conv = [system_msg] + pair
        else:
            full_conv = pair
        pair_text = tokenizer.apply_chat_template(
            full_conv, tokenize=True, add_generation_prompt=False, chat_template=chat_template
        )
        # Tokens for this pair = full - system
        pair_tokens.append(len(pair_text) - system_tokens)

    # Find how many pairs we can keep (from the start)
    # We want to keep the beginning of the conversation
    total_tokens = system_tokens
    keep_pairs = 0
    for tokens in pair_tokens:
        if total_tokens + tokens <= max_length:
            total_tokens += tokens
            keep_pairs += 1
        else:
            break

    # If we can't fit even one pair, drop the sample
    if keep_pairs == 0:
        return None

    # Reconstruct messages
    result_messages = []
    if system_msg:
        result_messages.append(system_msg)
    for pair in turn_pairs[:keep_pairs]:
        result_messages.extend(pair)

    # Check if the last message is from assistant (required for training)
    if not result_messages or result_messages[-1].get("role") != "assistant":
        return None

    return result_messages


def apply_truncation_strategy_to_example(
    example: dict[str, Any],
    tokenizer,
    max_length: int,
    strategy: str = "truncate",
    chat_template: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Apply truncation strategy to a single example.

    This is the main entry point for applying truncation strategies. It handles:
    - "truncate": Standard truncation with EOS removal
    - "drop": Returns None if sequence exceeds max_length
    - "split": Returns dict with "_split_chunks" key for post-processing
    - "truncate_turns": For conversational data, truncates by removing turn pairs

    Args:
        example: Dataset example with 'input_ids' and 'attention_mask' (tokenized)
                 or 'messages' (for truncate_turns before tokenization).
        tokenizer: Tokenizer for length calculation.
        max_length: Maximum sequence length.
        strategy: Truncation strategy to apply.
        chat_template: Optional chat template for truncate_turns.

    Returns:
        Modified example, or None if sample should be dropped.
        For "split", returns example with "_split_chunks" metadata.
    """
    if strategy not in VALID_TRUNCATION_STRATEGIES:
        raise ValueError(
            f"Invalid truncation_strategy: {strategy}. "
            f"Must be one of: {', '.join(VALID_TRUNCATION_STRATEGIES)}"
        )

    # Handle truncate_turns (operates on messages before tokenization)
    if strategy == "truncate_turns":
        if "messages" not in example:
            # Fall back to regular truncate for non-conversational data
            strategy = "truncate"
        else:
            truncated = truncate_conversation_by_turns(
                example["messages"],
                tokenizer,
                max_length,
                chat_template,
            )
            if truncated is None:
                return None
            result = dict(example)
            result["messages"] = truncated
            return result

    # For other strategies, we need tokenized data
    if "input_ids" not in example:
        return example  # Not tokenized yet, return as-is

    input_ids = example["input_ids"]
    attention_mask = example.get("attention_mask", [1] * len(input_ids))

    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id

    if strategy == "split":
        result = dict(example)
        # Ensure attention_mask is always present for consistency
        if "attention_mask" not in result:
            result["attention_mask"] = attention_mask

        if len(input_ids) <= max_length:
            # No splitting needed, but add empty _split_chunks for column consistency
            result["_split_chunks"] = None
            result["_split_mask_chunks"] = None
            return result

        # Get assistant_masks if present (for splitting alongside input_ids)
        assistant_masks = example.get("assistant_masks")

        chunks = split_tokens_into_chunks(
            input_ids, attention_mask, max_length, eos_token_id, bos_token_id
        )
        # Return first chunk and store rest for expansion
        result["input_ids"] = chunks[0][0]
        result["attention_mask"] = chunks[0][1]

        # Split assistant_masks if present
        if assistant_masks is not None:
            # Split assistant_masks using the same chunk boundaries
            mask_chunks = split_tokens_into_chunks(
                assistant_masks, attention_mask, max_length, eos_token_id, bos_token_id
            )
            result["assistant_masks"] = mask_chunks[0][0]
            # Store mask chunks alongside token chunks for expansion
            result["_split_mask_chunks"] = mask_chunks[1:] if len(mask_chunks) > 1 else None
        else:
            result["_split_mask_chunks"] = None

        result["_split_chunks"] = chunks[1:] if len(chunks) > 1 else None
        return result

    # truncate or drop
    truncated = truncate_tokens_with_strategy(
        input_ids, attention_mask, max_length, eos_token_id, bos_token_id, strategy
    )
    if truncated is None:
        return None

    result = dict(example)
    result["input_ids"] = truncated[0]
    result["attention_mask"] = truncated[1]
    truncated_len = len(truncated[0])

    # Also truncate labels if present
    if "labels" in result and len(result["labels"]) > truncated_len:
        result["labels"] = result["labels"][:truncated_len]

    # Also truncate assistant_masks if present
    if "assistant_masks" in result and result["assistant_masks"] is not None and len(result["assistant_masks"]) > truncated_len:
        result["assistant_masks"] = result["assistant_masks"][:truncated_len]

    return result


def expand_split_chunks(dataset: "Dataset") -> "Dataset":
    """
    Expand split chunks into separate dataset rows.

    After applying "split" strategy, examples may have "_split_chunks" metadata.
    This function expands those into separate rows.

    Processes in batches to avoid holding the entire expanded dataset in Python
    memory at once (which can cause OOM for large datasets with many chunks).

    Args:
        dataset: Dataset with potential "_split_chunks" columns.

    Returns:
        Dataset with chunks expanded into separate rows.
    """
    if "_split_chunks" not in dataset.column_names:
        return dataset

    from datasets import Dataset as HFDataset, concatenate_datasets

    has_mask_chunks = "_split_mask_chunks" in dataset.column_names
    batch_size = 5000
    batch_datasets = []
    current_batch = []

    for example in dataset:
        # Add the main example (first chunk is already in input_ids)
        row = {k: v for k, v in example.items() if not k.startswith("_split_")}

        # Ensure attention_mask is present for the first chunk
        if "attention_mask" not in row and "input_ids" in row:
            row["attention_mask"] = [1] * len(row["input_ids"])

        current_batch.append(row)

        # Add remaining chunks
        chunks = example.get("_split_chunks")
        mask_chunks = example.get("_split_mask_chunks") if has_mask_chunks else None

        if chunks:
            for i, (chunk_ids, chunk_mask) in enumerate(chunks):
                chunk_row = dict(row)
                chunk_row["input_ids"] = chunk_ids
                chunk_row["attention_mask"] = chunk_mask
                if "labels" in chunk_row:
                    # For split chunks, labels = input_ids (full sequence loss)
                    chunk_row["labels"] = chunk_ids
                # Handle assistant_masks chunks if present
                if mask_chunks and i < len(mask_chunks):
                    chunk_row["assistant_masks"] = mask_chunks[i][0]
                current_batch.append(chunk_row)

        # Flush batch to Arrow to free Python memory
        if len(current_batch) >= batch_size:
            batch_datasets.append(HFDataset.from_list(current_batch))
            current_batch = []

    # Flush remaining rows
    if current_batch:
        batch_datasets.append(HFDataset.from_list(current_batch))

    if not batch_datasets:
        return dataset
    if len(batch_datasets) == 1:
        return batch_datasets[0]

    return concatenate_datasets(batch_datasets)


# ---------------------------------------------------------------------------
# Standalone tokenization and truncation functions
# (Used by both loft prepare and SFTTrainer)
# ---------------------------------------------------------------------------


def mask_to_last_segment_only(mask: list[int]) -> list[int]:
    """
    Transform an assistant mask to only keep the last contiguous segment of 1s.

    Used for ``last_assistant_only_loss`` where we only want to train on the
    final assistant response in a multi-turn conversation.
    """
    if not mask or 1 not in mask:
        return mask

    result = [0] * len(mask)
    last_one_idx = None
    for i in range(len(mask) - 1, -1, -1):
        if mask[i] == 1:
            last_one_idx = i
            break
    if last_one_idx is None:
        return result
    start_idx = last_one_idx
    while start_idx > 0 and mask[start_idx - 1] == 1:
        start_idx -= 1
    for i in range(start_idx, last_one_idx + 1):
        result[i] = 1
    return result


def compute_assistant_mask_from_tokens(
    input_ids: list[int],
    processing_class,
) -> Optional[list[int]]:
    """
    Compute an assistant mask based on special tokens in the input.

    This is a fallback for chat templates that don't support the ``{% generation %}``
    macro. It looks for common assistant start/end token patterns:
    - <|assistant_start|> / <|assistant_end|>
    - <|im_start|>assistant / <|im_end|>
    - [/INST] / </s> (Llama-2 style)
    - <|start_header_id|>assistant / <|eot_id|> (Llama-3 style)

    Args:
        input_ids: Token IDs from the tokenized sequence.
        processing_class: Tokenizer to get special token IDs.

    Returns:
        A list of 0/1 where 1 = assistant token (train), 0 = non-assistant (mask).
        Returns None if no assistant tokens can be detected.
    """
    # Try to find assistant start/end token IDs from the tokenizer's vocabulary
    vocab = processing_class.get_vocab() if hasattr(processing_class, "get_vocab") else {}

    # Common patterns for assistant start tokens
    assistant_start_patterns = [
        "<|assistant_start|>",
        "<|assistant|>",
        "<|start_header_id|>",  # Llama-3 uses this followed by assistant text
    ]
    # Common patterns for assistant end tokens
    assistant_end_patterns = [
        "<|assistant_end|>",
        "<|eot_id|>",
        "<|im_end|>",
        "<|end|>",
    ]
    # Also check for tokens that mark other roles (to end assistant segments)
    role_start_patterns = [
        "<|user_start|>",
        "<|user|>",
        "<|system_start|>",
        "<|system|>",
        "<|developer_start|>",
        "<|tool_start|>",
    ]

    assistant_start_ids = set()
    assistant_end_ids = set()
    role_start_ids = set()

    for pattern in assistant_start_patterns:
        if pattern in vocab:
            assistant_start_ids.add(vocab[pattern])

    for pattern in assistant_end_patterns:
        if pattern in vocab:
            assistant_end_ids.add(vocab[pattern])

    for pattern in role_start_patterns:
        if pattern in vocab:
            role_start_ids.add(vocab[pattern])

    # If we can't find the special tokens, we can't compute the mask
    if not assistant_start_ids:
        return None

    # Build the mask
    mask = [0] * len(input_ids)
    in_assistant = False

    for i, token_id in enumerate(input_ids):
        if token_id in assistant_start_ids:
            # Start of assistant turn - mark AFTER this token
            in_assistant = True
            # Don't include the start token itself
        elif token_id in assistant_end_ids:
            # End of assistant turn - include up to but not including this token
            in_assistant = False
        elif token_id in role_start_ids:
            # Another role started, end assistant segment
            in_assistant = False
        elif in_assistant:
            mask[i] = 1

    # If we found at least some assistant tokens, return the mask
    if sum(mask) > 0:
        return mask

    return None

def compute_assistant_mask_from_messages(
    messages: list[dict[str, str]],
    tokenizer,
    **template_kwargs,
) -> list[int]:
    """
    Compute an assistant mask by tokenizing the conversation turn-by-turn.
    This is much more robust than token-matching as it uses the template logic.
    """
    # 1. Get the full tokenized sequence
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, **template_kwargs
    )
    mask = [0] * len(full_ids)

    # 2. Iterate through messages and find assistant turns
    for i, msg in enumerate(messages):
        role = msg.get("role") or msg.get("from", "")
        if role.lower() in ("assistant", "gpt"):
            # Find where this assistant turn starts
            # Prefix is everything before this message + the generation prompt (header)
            prefix_ids = tokenizer.apply_chat_template(
                messages[:i], tokenize=True, add_generation_prompt=True, **template_kwargs
            )
            # Find where this assistant turn ends
            # Context is everything up to and including this message
            full_turn_ids = tokenizer.apply_chat_template(
                messages[:i+1], tokenize=True, add_generation_prompt=False, **template_kwargs
            )
            
            start_idx = len(prefix_ids)
            end_idx = len(full_turn_ids)
            # Ensure we don't go out of bounds (templates can be tricky with EOS)
            for j in range(start_idx, min(end_idx, len(mask))):
                mask[j] = 1

    return mask

def remove_trailing_eos(input_ids: list[int], eos_token_id: int) -> list[int]:
    """Remove trailing EOS token(s) from *input_ids*."""
    while input_ids and input_ids[-1] == eos_token_id:
        input_ids = input_ids[:-1]
    return input_ids


def tokenize_sft_example(
    example: dict,
    processing_class,
    dataset_text_field: str = "text",
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    train_on_incomplete_assistant: bool = False,
    eos_token_id: Optional[int] = None,
) -> dict:
    """
    Tokenize a single SFT example (conversational or plain text).

    This is the standalone version of the tokenization logic that lives inside
    ``SFTTrainer._prepare_dataset``.  It handles:

    * Prompt-completion datasets (``prompt`` + ``completion`` columns)
    * Conversational datasets (``messages`` column)
    * Plain text datasets (``text`` or custom ``dataset_text_field``)

    Returns a dict with at least ``input_ids`` and optionally
    ``completion_mask`` and ``assistant_masks``.
    """
    need_assistant_masks = assistant_only_loss or last_assistant_only_loss

    # Check if last message is from assistant (for train_on_incomplete_assistant)
    last_role_is_assistant = False
    if train_on_incomplete_assistant:
        messages = example.get("messages") or (
            example.get("prompt", []) + example.get("completion", [])
        )
        if messages and isinstance(messages, list) and len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                role = last_msg.get("role") or last_msg.get("from", "")
                last_role_is_assistant = role.lower() in ("assistant", "gpt")

    if "prompt" in example:  # prompt-completion case
        output = {}
        if is_conversational(example):
            prompt_ids = processing_class.apply_chat_template(
                example["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
            prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
            prompt_completion_processed = processing_class.apply_chat_template(
                example["prompt"] + example["completion"],
                return_dict=True,
                tokenize=True,
                return_assistant_tokens_mask=need_assistant_masks,
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
            prompt_completion_processed = {
                k: v[0] if isinstance(v[0], list) else v
                for k, v in prompt_completion_processed.items()
            }
            prompt_completion_ids = prompt_completion_processed["input_ids"]
            if "assistant_masks" in prompt_completion_processed:
                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]

            # BETTER FALLBACK: If mask is missing or all zeros, compute from messages
            if need_assistant_masks:
                asst_masks = output.get("assistant_masks", [])
                if not asst_masks or sum(asst_masks) == 0:
                    # Use the new robust message-based computation
                    output["assistant_masks"] = compute_assistant_mask_from_messages(
                        example["messages"],
                        processing_class,
                        **example.get("chat_template_kwargs", {})
                    )
        else:  # plain text prompt-completion case
            prompt_ids = processing_class(example["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = processing_class(example["completion"], add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + completion_ids
            output["input_ids"] = input_ids
            output["completion_mask"] = [0] * len(prompt_ids) + [1] * len(completion_ids)
            if need_assistant_masks:
                # Treat completion as assistant content for mask consistency
                output["assistant_masks"] = [0] * len(prompt_ids) + [1] * len(completion_ids)

    else:  # language modeling case
        if is_conversational(example):
            processed = processing_class.apply_chat_template(
                example["messages"],
                return_dict=True,
                tokenize=True,
                return_assistant_tokens_mask=need_assistant_masks,
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
            processed = {k: v[0] if isinstance(v[0], list) else v for k, v in processed.items()}
            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}

            # Fallback: if mask is missing or all zeros, compute from messages
            if need_assistant_masks:
                asst_masks = output.get("assistant_masks", [])
                if not asst_masks or sum(asst_masks) == 0:
                    # Use the new robust message-based computation
                    output["assistant_masks"] = compute_assistant_mask_from_messages(
                        example["messages"],
                        processing_class,
                        **example.get("chat_template_kwargs", {})
                    )
        else:  # plain text case
            text = example.get(dataset_text_field, "")
            input_ids = processing_class(text, add_special_tokens=False)["input_ids"]
            output = {"input_ids": input_ids}
            # Plain text has no roles, so train on all tokens
            if need_assistant_masks:
                output["assistant_masks"] = [1] * len(input_ids)

    # Apply last_assistant_only_loss: mask all but the last assistant turn
    if last_assistant_only_loss and "assistant_masks" in output:
        output["assistant_masks"] = mask_to_last_segment_only(output["assistant_masks"])

    # Apply train_on_incomplete_assistant: remove trailing EOS if last role is assistant
    if train_on_incomplete_assistant and last_role_is_assistant and eos_token_id is not None:
        output["input_ids"] = remove_trailing_eos(output["input_ids"], eos_token_id)
        if "assistant_masks" in output:
            output["assistant_masks"] = output["assistant_masks"][: len(output["input_ids"])]
        if "completion_mask" in output:
            output["completion_mask"] = output["completion_mask"][: len(output["input_ids"])]

    return output


def apply_truncation_to_dataset(
    dataset: "Dataset",
    processing_class,
    max_length: int,
    strategy: str = "truncate",
    num_proc: Optional[int] = None,
) -> "Dataset":
    """
    Apply a truncation strategy to a tokenized dataset.

    Args:
        dataset: HF Dataset with ``input_ids`` column.
        processing_class: Tokenizer (used by split/truncate internals).
        max_length: Maximum sequence length (default; can be overridden per-example).
        strategy: One of ``"truncate"``, ``"drop"``, ``"split"``.
            (``"truncate_turns"`` should be applied *before* tokenization.)
        num_proc: Number of processes for ``dataset.map``.

    Returns:
        Dataset with truncation applied.  For ``"split"``, chunks are expanded
        into separate rows.

    Note:
        If the dataset has a ``_max_length`` column, per-example max_length overrides
        are used. This allows different datasets in a prepare to have different truncation
        lengths (e.g., 2048 for short-form data while training at 4096 context).

        If the dataset has a ``_truncation_strategy`` column, per-example strategy
        overrides are used. This allows different datasets in a prepare to have different
        truncation strategies (e.g., split for prose, truncate for chat).
    """
    import logging as _logging
    from datasets import concatenate_datasets

    _logger = _logging.getLogger(__name__)

    map_kwargs: dict = {}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc

    if strategy == "truncate_turns":
        strategy = "truncate"  # already handled pre-tokenization

    # Check if per-example settings are available
    has_per_example_max_length = "_max_length" in dataset.column_names
    has_per_example_strategy = "_truncation_strategy" in dataset.column_names

    # If we have per-example strategies, split dataset by strategy and process each separately
    if has_per_example_strategy:
        # Get unique strategies in the dataset
        strategies_in_data = set(dataset["_truncation_strategy"])
        # Map truncate_turns to truncate (already handled pre-tokenization)
        strategies_in_data = {s if s != "truncate_turns" else "truncate" for s in strategies_in_data if s is not None}
        # Add None -> use default strategy
        if None in set(dataset["_truncation_strategy"]):
            strategies_in_data.add(strategy)

        result_datasets = []
        for strat in strategies_in_data:
            # Filter to samples with this strategy (or None -> default)
            if strat == strategy:
                # Include samples with this strategy OR with None (default)
                subset_indices = [
                    i for i, s in enumerate(dataset["_truncation_strategy"])
                    if s == strat or s is None or (s == "truncate_turns" and strat == "truncate")
                ]
            else:
                subset_indices = [
                    i for i, s in enumerate(dataset["_truncation_strategy"])
                    if s == strat or (s == "truncate_turns" and strat == "truncate")
                ]

            if not subset_indices:
                continue

            subset = dataset.select(subset_indices)
            # Remove the _truncation_strategy column before processing
            subset = subset.remove_columns(["_truncation_strategy"])

            _logger.info(f"Applying {strat} strategy to {len(subset):,} samples")

            # Recursively apply truncation with the specific strategy
            processed = apply_truncation_to_dataset(
                subset, processing_class, max_length, strategy=strat, num_proc=num_proc
            )
            result_datasets.append(processed)

        if result_datasets:
            return concatenate_datasets(result_datasets)
        return dataset

    # Standard path: single strategy for all samples
    if strategy == "drop":
        original_len = len(dataset)
        if has_per_example_max_length:
            # Use per-example max_length if available
            dataset = dataset.filter(
                lambda x: len(x.get("input_ids", [])) <= x.get("_max_length", max_length),
                num_proc=num_proc,
            )
        else:
            dataset = dataset.filter(
                lambda x: len(x.get("input_ids", [])) <= max_length,
                num_proc=num_proc,
            )
        filtered_len = len(dataset)
        if filtered_len < original_len:
            _logger.info(
                f"drop strategy: Filtered out {original_len - filtered_len} samples "
                f"exceeding max_length."
            )

    elif strategy == "split":
        def _split_and_expand_batch(examples, tokenizer, default_max_length):
            """Split oversized sequences into max_length chunks, expanding rows inline.

            Uses batched map so that one input row can produce multiple output rows
            without needing a separate expansion pass (which previously required
            holding the entire expanded dataset in Python memory).
            """
            eos_id = tokenizer.eos_token_id
            bos_id = tokenizer.bos_token_id
            n = len(examples["input_ids"])
            has_attn = "attention_mask" in examples
            has_masks = "assistant_masks" in examples
            has_labels = "labels" in examples
            has_per_max = "_max_length" in examples

            # Output columns: everything from input minus _max_length, ensure attention_mask
            skip_cols = {"_max_length"}
            out_keys = [k for k in examples if k not in skip_cols]
            if "attention_mask" not in out_keys:
                out_keys.append("attention_mask")
            result = {k: [] for k in out_keys}

            token_cols = {"input_ids", "attention_mask", "assistant_masks", "labels"}
            meta_keys = [k for k in out_keys if k not in token_cols]

            for i in range(n):
                input_ids = examples["input_ids"][i]
                attn_mask = examples["attention_mask"][i] if has_attn else [1] * len(input_ids)
                eff_max = default_max_length
                if has_per_max and examples["_max_length"][i] is not None:
                    eff_max = examples["_max_length"][i]

                if len(input_ids) <= eff_max:
                    # No split needed — emit single row
                    result["input_ids"].append(input_ids)
                    result["attention_mask"].append(attn_mask)
                    if has_masks:
                        result["assistant_masks"].append(examples["assistant_masks"][i])
                    if has_labels:
                        result["labels"].append(examples["labels"][i])
                    for k in meta_keys:
                        result[k].append(examples[k][i])
                    continue

                # Split into chunks
                chunks = split_tokens_into_chunks(input_ids, attn_mask, eff_max, eos_id, bos_id)
                mask_chunks = None
                if has_masks and examples["assistant_masks"][i] is not None:
                    mask_chunks = split_tokens_into_chunks(
                        examples["assistant_masks"][i], attn_mask, eff_max, eos_id, bos_id
                    )

                for ci, (cids, cmask) in enumerate(chunks):
                    result["input_ids"].append(cids)
                    result["attention_mask"].append(cmask)
                    if has_labels:
                        result["labels"].append(cids)  # split chunks: labels = input_ids
                    if has_masks:
                        if mask_chunks and ci < len(mask_chunks):
                            result["assistant_masks"].append(mask_chunks[ci][0])
                        else:
                            result["assistant_masks"].append(None)
                    for k in meta_keys:
                        result[k].append(examples[k][i])

            return result

        dataset = dataset.map(
            _split_and_expand_batch,
            fn_kwargs={"tokenizer": processing_class, "default_max_length": max_length},
            batched=True,
            batch_size=1000,
            desc="Splitting into chunks",
            **map_kwargs,
        )

    else:  # "truncate"
        def _apply_truncate(example, tokenizer, default_max_length):
            # Use per-example max_length if present, otherwise use default
            effective_max_length = example.pop("_max_length", None) or default_max_length
            return apply_truncation_strategy_to_example(
                example, tokenizer, effective_max_length, strategy="truncate"
            )

        dataset = dataset.map(
            _apply_truncate,
            fn_kwargs={"tokenizer": processing_class, "default_max_length": max_length},
            remove_columns=["_max_length"] if has_per_example_max_length else None,
            desc="Truncating",
            **map_kwargs,
        )

    # Clean up _max_length column if it's still present (e.g., for drop strategy)
    if "_max_length" in dataset.column_names:
        dataset = dataset.remove_columns(["_max_length"])

    return dataset
