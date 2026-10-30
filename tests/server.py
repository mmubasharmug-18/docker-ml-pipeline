import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_env(**settings) -> dict:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "APP_HOST": "127.0.0.1",
        **{key: str(value) for key, value in settings.items()},
    }


class Server:
    def __init__(self, port: int, process: subprocess.Popen):
        self.port = port
        self.process = process
        self.output = ""

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def request(self, path: str, body: bytes = None):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = urllib.request.Request(self.url + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()


def wait_until_up(server: Server, seconds: float = 30) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError("server exited early:\n" + server.process.stdout.read())
        try:
            server.request("/health")
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("server did not start in time")


@contextmanager
def running_server(**settings):
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "serving"],
        env=start_env(APP_PORT=port, **settings),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server = Server(port, process)
    try:
        wait_until_up(server)
        yield server
    finally:
        process.terminate()
        try:
            server.output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            server.output, _ = process.communicate()
