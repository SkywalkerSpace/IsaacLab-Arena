from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import logging
import time
from typing import Any, Callable

import msgpack
import numpy as np
import zmq

from gr00t.data.types import ModalityConfig
from gr00t.data.utils import to_json_serializable

from .policy import BasePolicy


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": to_json_serializable(obj)}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str = None,
        access_log: bool = False,
    ):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token
        self.access_log = access_log
        self._request_count = 0

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            getattr(self.policy, "get_modality_config", lambda: {}),
            requires_input=False,
        )

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def _log_access(self, event: str, **fields: Any) -> None:
        """Print one flushed JSON record without request payloads or credentials."""
        if self.access_log:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "request_id": self._request_count,
                "event": event,
                **fields,
            }
            print("[access] " + json.dumps(record, ensure_ascii=True), flush=True)

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            started_at = None
            endpoint = None
            status = "error"
            try:
                message = self.socket.recv()
                started_at = time.perf_counter()
                self._request_count += 1
                self._log_access("received", request_bytes=len(message))
                request = MsgSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")

                # Validate token before processing request
                if not self._validate_token(request):
                    status = "unauthorized"
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(**request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(MsgSerializer.to_bytes(result))
                status = "ok"
            except Exception as e:
                status = "error"
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))
            finally:
                if started_at is not None:
                    self._log_access(
                        "completed",
                        endpoint=endpoint[:128] if isinstance(endpoint, str) else "<invalid>",
                        status=status,
                        request_bytes=len(message),
                        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
                    )

    @staticmethod
    def start_server(policy: BasePolicy, port: int, api_token: str = None):
        server = PolicyServer(policy, port=port, api_token=api_token)
        server.run()


class PolicyClient(BasePolicy):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,
        max_retries: int = 0,
        retry_backoff_sec: float = 1.0,
    ):
        """Create a client with bounded timeout retries (at-least-once delivery).

        Leave max_retries at zero for stateful policies such as ReplayPolicy:
        a timeout can mean the response was lost after the request was executed.
        """
        super().__init__(strict=strict)
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not 0 <= retry_backoff_sec < float("inf"):
            raise ValueError("retry_backoff_sec must be finite and non-negative")
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        old_socket = getattr(self, "socket", None)
        if old_socket is not None:
            old_socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.Again:
            # call_endpoint already recreated the timed-out socket.
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        payload = MsgSerializer.to_bytes(request)
        for attempt in range(self.max_retries + 1):
            try:
                self.socket.send(payload)
                message = self.socket.recv()
                break
            except zmq.error.Again:
                # A REQ socket cannot send again while waiting for a reply.
                # Drop pending messages and create a new socket even on the
                # last attempt, so a later caller can safely use this client.
                self._init_socket()
                if attempt == self.max_retries:
                    logging.error(
                        "GR00T %s at %s:%s timed out after %d attempt(s) "
                        "(send/receive timeout %d ms)",
                        endpoint,
                        self.host,
                        self.port,
                        attempt + 1,
                        self.timeout_ms,
                    )
                    raise
                delay = self.retry_backoff_sec * (2 ** min(attempt, 6))
                logging.warning(
                    "GR00T %s at %s:%s timed out; retry %d/%d in %.1f s. "
                    "The previous request may already have executed.",
                    endpoint,
                    self.host,
                    self.port,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        socket = getattr(self, "socket", None)
        if socket is not None:
            socket.close(linger=0)
        context = getattr(self, "context", None)
        if context is not None:
            context.term()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )
