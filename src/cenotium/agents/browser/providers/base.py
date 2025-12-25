"""Base LLM provider classes for browser agent."""

import base64
import io
import json
import re

from anthropic import Anthropic
from openai import OpenAI
from PIL import Image


def Message(content, role="assistant"):
    return {"role": role, "content": content}


def Text(text):
    return {"type": "text", "text": text}


def parse_json(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


class LLMProvider:
    """Base LLM provider with tool use support."""

    base_url = None
    api_key = None
    aliases = {}

    def __init__(self, model):
        self.model = self.aliases.get(model, model)
        self.client = self.create_client()

    def create_function_schema(self, definitions):
        functions = []
        for name, details in definitions.items():
            properties = {}
            required = []
            for param_name, param_desc in details["params"].items():
                properties[param_name] = {"type": "string", "description": param_desc}
                required.append(param_name)
            function_def = self.create_function_def(name, details, properties, required)
            functions.append(function_def)
        return functions

    def create_tool_call(self, name, parameters):
        return {"type": "function", "name": name, "parameters": parameters}

    def wrap_block(self, block):
        if isinstance(block, bytes):
            return self.create_image_block(block)
        return Text(block)

    def transform_message(self, message):
        content = message["content"]
        if isinstance(content, list):
            wrapped_content = [self.wrap_block(block) for block in content]
            return {**message, "content": wrapped_content}
        return message

    def completion(self, messages, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        new_messages = [self.transform_message(message) for message in messages]
        completion = self.client.create(
            messages=new_messages, model=self.model, **filtered_kwargs
        )
        if hasattr(completion, "error"):
            raise Exception(f"Error calling model: {completion.error}")
        return completion


class OpenAIBaseProvider(LLMProvider):
    """OpenAI-compatible LLM provider."""

    def create_client(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key).chat.completions

    def create_function_def(self, name, details, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": details["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def create_image_block(self, image_data: bytes):
        image_type = "png"
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                image_type = img.format.lower()
        except Exception:
            pass
        encoded = base64.b64encode(image_data).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_type};base64,{encoded}"},
        }

    def call(self, messages, functions=None):
        tools = self.create_function_schema(functions) if functions else None
        completion = self.completion(messages, tools=tools)
        message = completion.choices[0].message

        if functions:
            tool_calls = message.tool_calls or []
            combined_tool_calls = [
                self.create_tool_call(
                    tool_call.function.name, parse_json(tool_call.function.arguments)
                )
                for tool_call in tool_calls
                if parse_json(tool_call.function.arguments) is not None
            ]

            if message.content and not tool_calls:
                tool_call_matches = re.search(r"\{.*\}", message.content)
                if tool_call_matches:
                    tool_call = parse_json(tool_call_matches.group(0))
                    parameters = tool_call.get("parameters", tool_call.get("arguments"))
                    if tool_call.get("name") and parameters:
                        combined_tool_calls.append(
                            self.create_tool_call(tool_call.get("name"), parameters)
                        )
                        return None, combined_tool_calls

            return message.content, combined_tool_calls

        return message.content


class AnthropicBaseProvider(LLMProvider):
    """Anthropic LLM provider."""

    def create_client(self):
        return Anthropic(api_key=self.api_key).messages

    def create_function_def(self, name, details, properties, required):
        return {
            "name": name,
            "description": details["description"],
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def create_image_block(self, base64_image):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_image,
            },
        }

    def call(self, messages, functions=None):
        tools = self.create_function_schema(functions) if functions else None
        system = "\n".join(
            msg.get("content") for msg in messages if msg.get("role") == "system"
        )
        messages = [msg for msg in messages if msg.get("role") != "system"]
        completion = self.completion(
            messages, system=system, tools=tools, max_tokens=4096
        )
        text = "".join(getattr(block, "text", "") for block in completion.content)

        if functions:
            tool_calls = [
                self.create_tool_call(block.name, block.input)
                for block in completion.content
                if block.type == "tool_use"
            ]
            return text, tool_calls

        return text


class MistralBaseProvider(OpenAIBaseProvider):
    """Mistral LLM provider."""

    def create_function_def(self, name, details, properties, required):
        if isinstance(details.get("description"), dict):
            details["description"] = details["description"].get("description", "")
        return super().create_function_def(name, details, properties, required)

    def call(self, messages, functions=None):
        if messages and messages[-1].get("role") == "assistant":
            prefix = messages.pop()["content"]
            if messages and messages[-1].get("role") == "user":
                messages[-1]["content"] = (
                    prefix + "\n" + messages[-1].get("content", "")
                )
            else:
                messages.append({"role": "user", "content": prefix})
        return super().call(messages, functions)
