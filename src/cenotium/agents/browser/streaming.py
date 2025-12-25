"""Streaming and display utilities for browser agent."""

import asyncio
import os
import signal
import subprocess
import sys


class DisplayClient:
    """Client to view and save a live display stream."""

    def __init__(self, output_dir: str = "."):
        self.process = None
        self.output_stream = f"{output_dir}/output.ts"
        self.output_file = f"{output_dir}/output.mp4"

    async def start(self, stream_url: str, title: str = "Sandbox", delay: int = 0):
        title = title.replace("'", "\\'")
        self.process = await asyncio.create_subprocess_shell(
            f"sleep {delay} && ffmpeg -reconnect 1 -i {stream_url} -c:v libx264 -preset fast -crf 23 "
            f"-c:a aac -b:a 128k -f mpegts -loglevel quiet - | tee {self.output_stream} | "
            f"ffplay -autoexit -i -loglevel quiet -window_title '{title}' -",
            preexec_fn=os.setsid,
            stdin=asyncio.subprocess.DEVNULL,
        )

    async def stop(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            await self.process.wait()

    async def save_stream(self):
        process = await asyncio.create_subprocess_shell(
            f"ffmpeg -i {self.output_stream} -c:v copy -c:a copy -loglevel quiet {self.output_file}"
        )
        await process.wait()
        if process.returncode == 0:
            print(f"Stream saved as {self.output_file}")
        else:
            print(f"Failed to save stream as {self.output_file}")


class Browser:
    """Client to show a VNC client to the sandbox."""

    def __init__(self, port: int):
        self.process = None
        self.port = port

    def start(self, stream_url: str, title: str = "Sandbox"):
        script_path = os.path.join(os.path.dirname(__file__), "browser_script.py")
        try:
            self.process = subprocess.Popen(
                [sys.executable, script_path, stream_url, title, str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Failed to start browser: {e}")

    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process = None
            except Exception as e:
                print(f"Failed to stop browser: {e}")
