"""OS-Atlas grounding model provider."""

import logging
import os

from gradio_client import Client, handle_file

from ..grounding import extract_bbox_midpoint

logger = logging.getLogger(__name__)

OSATLAS_HUGGINGFACE_SOURCE = "maxiw/OS-ATLAS"
OSATLAS_HUGGINGFACE_MODEL = "OS-Copilot/OS-Atlas-Base-7B"
OSATLAS_HUGGINGFACE_API = "/run_example"

HF_TOKEN = os.getenv("HF_TOKEN")


class OSAtlasProvider:
    """Provider for OS-Atlas grounding model."""

    def __init__(self):
        self.client = Client(OSATLAS_HUGGINGFACE_SOURCE, hf_token=HF_TOKEN)

    def call(self, prompt, image_data):
        result = self.client.predict(
            image=handle_file(image_data),
            text_input=prompt + "\nReturn the response in the form of a bbox",
            model_id=OSATLAS_HUGGINGFACE_MODEL,
            api_name=OSATLAS_HUGGINGFACE_API,
        )
        position = extract_bbox_midpoint(result[1])
        logger.debug(f"bbox {result[2]}")
        return position
