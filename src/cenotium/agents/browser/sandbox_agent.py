"""Sandbox agent for browser automation."""

import json
import logging
import os
import tempfile
from typing import Dict, Optional

from PIL import Image

from .grounding import draw_big_dot
from .providers.base import Message

logger = logging.getLogger(__name__)

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

TOOLS = {
    "stop": {
        "description": "Indicate that the task has been completed.",
        "params": {},
    }
}


def tool(description, params):
    """Decorator to register a tool."""

    def decorator(func):
        TOOLS[func.__name__] = {"description": description, "params": params}
        return func

    return decorator


class SandboxAgent:
    """Agent that operates within a sandbox environment."""

    def __init__(
        self,
        sandbox,
        vision_model,
        action_model,
        grounding_model,
        output_dir: str = ".",
        additional_context: str = "",
    ):
        self.sandbox = sandbox
        self.vision_model = vision_model
        self.action_model = action_model
        self.grounding_model = grounding_model
        self.messages = []
        self.latest_screenshot = None
        self.image_counter = 0
        self.tmp_dir = tempfile.mkdtemp()
        self.additional_context = additional_context

    def call_function(self, name: str, arguments: Optional[Dict] = None):
        func_impl = getattr(self, name.lower(), None) if name.lower() in TOOLS else None
        if func_impl:
            try:
                return func_impl(**arguments) if arguments else func_impl()
            except Exception as e:
                return f"Error: {e}"
        return "Function not implemented."

    def save_image(self, image, prefix: str = "image") -> str:
        self.image_counter += 1
        filename = f"{prefix}_{self.image_counter}.png"
        filepath = os.path.join(self.tmp_dir, filename)
        if isinstance(image, Image.Image):
            image.save(filepath)
        else:
            with open(filepath, "wb") as f:
                f.write(image)
        return filepath

    def take_screenshot(self) -> bytes:
        file = self.sandbox.take_screenshot()
        filename = self.save_image(file, "screenshot")
        logger.debug(f"screenshot {filename}")
        self.latest_screenshot = filename
        with open(filename, "rb") as image_file:
            return image_file.read()

    @tool(
        description="Run a shell command and return the result.",
        params={"command": "Shell command to run synchronously"},
    )
    def run_command(self, command: str) -> str:
        result = self.sandbox.commands.run(command, timeout=5)
        stdout, stderr = result.stdout, result.stderr
        if stdout and stderr:
            return stdout + "\n" + stderr
        return stdout + stderr if stdout or stderr else "Command completed."

    @tool(
        description="Run a shell command in the background.",
        params={"command": "Shell command to run asynchronously"},
    )
    def run_background_command(self, command: str) -> str:
        self.sandbox.commands.run(command, background=True)
        return "Command started."

    @tool(
        description="Send a key or combination of keys to the system.",
        params={"name": "Key or combination (e.g. 'Return', 'Ctrl-C')"},
    )
    def send_key(self, name: str) -> str:
        self.sandbox.press(name)
        return "Key pressed."

    @tool(
        description="Type a specified text into the system.",
        params={"text": "Text to type"},
    )
    def type_text(self, text: str) -> str:
        self.sandbox.write(
            text, chunk_size=TYPING_GROUP_SIZE, delay_in_ms=TYPING_DELAY_MS
        )
        return "Text typed."

    def click_element(self, query: str, click_command, action_name: str = "click"):
        self.take_screenshot()
        position = self.grounding_model.call(query, self.latest_screenshot)
        dot_image = draw_big_dot(Image.open(self.latest_screenshot), position)
        self.save_image(dot_image, "location")

        x, y = position
        self.sandbox.move_mouse(x, y)
        click_command()
        return f"Mouse {action_name}ed."

    @tool(
        description="Click on a specified UI element.",
        params={"query": "UI element to click"},
    )
    def click(self, query: str) -> str:
        return self.click_element(query, self.sandbox.left_click)

    @tool(
        description="Double click on a specified UI element.",
        params={"query": "UI element to double click"},
    )
    def double_click(self, query: str) -> str:
        return self.click_element(query, self.sandbox.double_click, "double click")

    @tool(
        description="Right click on a specified UI element.",
        params={"query": "UI element to right click"},
    )
    def right_click(self, query: str) -> str:
        return self.click_element(query, self.sandbox.right_click, "right click")

    def append_screenshot(self) -> str:
        return self.vision_model.call(
            [
                *self.messages,
                Message(
                    [
                        self.take_screenshot(),
                        "This image shows the current display. Please respond:\n"
                        "Objective: [state the objective]\n"
                        "On screen: [list relevant elements]\n"
                        "Status: [complete|not complete]\n\n"
                        "(If not complete:)\n"
                        "Next step: [action] [details] to [expected result].",
                    ],
                    role="user",
                ),
            ]
        )

    async def run(self, instruction: str, context: Optional[str] = None) -> str:
        self.messages.append(Message(f"OBJECTIVE: {instruction}"))
        logger.info(f"USER: {instruction}")

        should_continue = True
        while should_continue:
            self.sandbox.set_timeout(600)
            thought = self.append_screenshot()
            logger.info(f"THOUGHT: {thought}")

            content, tool_calls = self.action_model.call(
                [
                    Message(
                        "You are an AI assistant with computer use abilities.",
                        role="system",
                    ),
                    *self.messages,
                    Message(thought),
                    Message("Use tool calls to take actions, or stop if complete."),
                ],
                TOOLS,
            )

            if content:
                self.messages.append(Message(content))
                logger.info(f"THOUGHT: {content}")

            should_continue = False
            for tool_call in tool_calls:
                name = tool_call.get("name")
                parameters = tool_call.get("parameters")
                should_continue = name != "stop"
                if not should_continue:
                    break

                logger.info(f"ACTION: {name} {parameters}")
                self.messages.append(Message(json.dumps(tool_call)))
                result = self.call_function(name, parameters)
                self.messages.append(Message(f"OBSERVATION: {result}"))
                logger.info(f"OBSERVATION: {result}")

        return "Task completed"
