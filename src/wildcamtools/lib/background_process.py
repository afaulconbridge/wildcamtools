import contextlib
import subprocess
import time
from collections.abc import Callable, Iterable


class BackgroundProcess:
    """
    Usage:
        with BackgroundProcess(['myserver', '--flag'], cwd='/path', ready_check=lambda: port_open(8000)):
            # process is running here
            do_work()
    """

    def __init__(
        self,
        cmd: Iterable[str],
        *,
        cwd: str | None = None,
        env: dict | None = None,
        stdout=None,
        stderr=None,
        start_wait: float = 15.0,
        ready_check: Callable[[], bool] | None = None,
        ready_timeout: float = 10.0,
        terminate_timeout: float = 5.0,
    ):
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env = env
        self.stdout = stdout
        self.stderr = stderr
        self.start_wait = start_wait
        self.ready_check = ready_check
        self.ready_timeout = ready_timeout
        self.terminate_timeout = terminate_timeout
        self.process: subprocess.Popen | None = None

    def _create_process(self):
        self.process = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=self.env,
            stdout=self.stdout,
            stderr=self.stderr,
            text=True,
        )

    def _check_till_ready(self):
        if self.ready_check:
            deadline = time.time() + self.ready_timeout
            while time.time() < deadline:
                if self.process.poll() is not None:
                    # process exited early
                    raise RuntimeError(f"Process exited prematurely with code {self.process.returncode}")
                if self.ready_check():
                    break
                time.sleep(self.start_wait)
            else:
                # timeout
                self._terminate_process()
                raise TimeoutError("Process did not become ready within timeout")

    def __enter__(self):
        # Start process
        self._create_process()

        # simple brief pause to give process time to start
        time.sleep(self.start_wait)

        # Optional readiness polling
        self._check_till_ready()

        return self

    def __exit__(self, exc_type, exc, tb):
        self._terminate_process()
        # Do not suppress exceptions
        return False

    def _terminate_process(self):
        if not self.process:
            return
        if self.process.poll() is not None:
            return  # already exited

        # graceful exit
        self.process.terminate()
        try:
            self.process.wait(timeout=self.terminate_timeout)
        except subprocess.TimeoutExpired:
            # force kill
            with contextlib.suppress(Exception):
                self.process.kill()
                self.process.wait(timeout=1)
