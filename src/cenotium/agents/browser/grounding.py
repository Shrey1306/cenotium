"""Grounding utilities for browser automation."""

import re

from PIL import ImageDraw


def draw_big_dot(image, coordinates, color="red", radius=12):
    """Draw a large dot on an image at the given coordinates."""
    draw = ImageDraw.Draw(image)
    x, y = coordinates
    bounding_box = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bounding_box, fill=color, outline=color)
    return image


def extract_bbox_midpoint(bbox_response):
    """Extract the midpoint from a bounding box response."""
    match = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", bbox_response)
    inner_text = match.group(1) if match else bbox_response
    numbers = [float(num) for num in re.findall(r"\d+\.\d+|\d+", inner_text)]
    if len(numbers) == 2:
        return numbers[0], numbers[1]
    elif len(numbers) >= 4:
        return (numbers[0] + numbers[2]) // 2, (numbers[1] + numbers[3]) // 2
    return None
