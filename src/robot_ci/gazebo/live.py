"""Live Gazebo -> browser bridge for the pick-and-place demo.

The demo runs Gazebo headless inside the container and shows it in a browser on
the Mac. Nothing here invents data: joint angles come off Gazebo's own
``joint_state`` topic and camera frames off Gazebo's own sensor topic, both as
produced by the running simulation.

Two transports are used, and which is used for what is forced rather than
chosen:

* **Camera** goes through ``gz.transport13`` directly. The Python bindings ship
  ``gz.msgs10.image_pb2``, so frames can be decoded in-process.
* **Joint state** goes through the ``gz`` CLI (``gz topic -e``). The vendored
  ``gz_msgs10`` in the ROS 2 Jazzy image does **not** contain
  ``joint_state.proto``, and because ``ros_gz_bridge`` links the same vendored
  package it has no template specialisation for ``gz.msgs.JointState`` either --
  bridging that pair fails with "No template specialization for the pair".
  Parsing the CLI's JSON output is therefore the available path, not a shortcut.

The bridge also serves the viewer itself and a ``/healthz`` endpoint, so
"the demo is ready" is a statement about observed topics and a decoded frame
rather than about a process having started.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

__all__ = [
    "JOINT_NAMES",
    "BridgeHealth",
    "CameraReader",
    "DemoContext",
    "JointStateReader",
    "LiveBridge",
    "robot_kinematics",
]

#: Joint order as the Panda model defines it. Fixed so the viewer and the
#: reader agree on ordering even before the first message arrives.
JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "left_finger",
    "right_finger",
)


@dataclass
class BridgeHealth:
    """What the demo has actually observed, for ``/healthz``.

    Each flag is set by evidence -- a topic that produced a message, a frame
    that decoded -- so a green light means data arrived.
    """

    gazebo_started: bool = False
    panda_loaded: bool = False
    joint_stream: bool = False
    camera_stream: bool = False
    viewer_served: bool = False
    joint_messages: int = 0
    camera_frames: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the health report as JSON-serialisable data."""
        return {
            "gazebo_started": self.gazebo_started,
            "panda_loaded": self.panda_loaded,
            "joint_stream": self.joint_stream,
            "camera_stream": self.camera_stream,
            "viewer_served": self.viewer_served,
            "joint_messages": self.joint_messages,
            "camera_frames": self.camera_frames,
            "errors": self.errors,
        }

    def ready(self) -> bool:
        """Whether every live stream is confirmed.

        The viewer counts as ready only once the page has been served, so the
        report cannot claim a demo that no one can open.
        """
        return self.panda_loaded and self.joint_stream and self.camera_stream and self.viewer_served


class JointStateReader:
    """Stream joint angles out of Gazebo's ``joint_state`` topic.

    The transport is the ``gz`` CLI, because the Python bindings in this image
    cannot decode ``gz.msgs.JointState`` (see the module docstring). The CLI is
    asked for JSON; a brace-matching reader reassembles the pretty-printed
    objects into whole documents, since the stream is not line-delimited.
    """

    def __init__(self, topic: str, health: BridgeHealth) -> None:
        self._topic = topic
        self._health = health
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._positions: dict[str, float] = dict.fromkeys(JOINT_NAMES, 0.0)
        self._sim_time = 0.0
        self._stop = threading.Event()

    def latest(self) -> tuple[dict[str, float], float]:
        """Return the most recent (positions, simulation time)."""
        with self._lock:
            return dict(self._positions), self._sim_time

    def start(self) -> None:
        """Begin reading on a background thread."""
        threading.Thread(target=self._run, name="gz-joint-reader", daemon=True).start()

    def stop(self) -> None:
        """Ask the reader to finish and terminate the CLI process."""
        self._stop.set()
        if self._process is not None:
            self._process.terminate()

    def _run(self) -> None:
        cmd = ["gz", "topic", "-e", "-t", self._topic, "--json-output"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:  # pragma: no cover - gz absent
            self._health.errors.append(f"joint reader could not start: {exc}")
            return

        buffer: list[str] = []
        depth = 0
        assert self._process.stdout is not None
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            depth += line.count("{") - line.count("}")
            buffer.append(line)
            if depth > 0:
                continue
            if buffer:
                self._consume("".join(buffer))
                buffer = []
        self._process.terminate()

    def _consume(self, document: str) -> None:
        try:
            payload = json.loads(document)
        except json.JSONDecodeError:
            return
        # JointState nests the truth: each entry in "joint" carries its own
        # name, and the angle lives under axis1 (axis2, unused by the Panda,
        # holds a second actuation axis). Older code assumed a flat
        # name/position layout; Gazebo emits only the nested one, so that flat
        # read silently reported zeros and must not be assumed again.
        entries = payload.get("joint")
        if isinstance(entries, list) and entries:
            with self._lock:
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name")
                    axis = entry.get("axis1") or {}
                    position = axis.get("position") if isinstance(axis, dict) else None
                    if name in self._positions and position is not None:
                        self._positions[name] = float(position)
                stamp = (payload.get("header") or {}).get("stamp") or {}
                if isinstance(stamp, dict) and "sec" in stamp:
                    self._sim_time = float(stamp["sec"]) + float(stamp.get("nsec", 0)) / 1e9
                self._health.joint_messages += 1
                self._health.joint_stream = True
                # Receiving a joint-state message is the proof that Gazebo is
                # up, so report it as such rather than leaving a flag that reads
                # false while a simulation is plainly running.
                self._health.gazebo_started = True
            return
        with self._lock:
            for name, position in zip(
                payload.get("name", []), payload.get("position", []), strict=False
            ):
                if name in self._positions:
                    self._positions[name] = float(position)
            stamp = (payload.get("header") or {}).get("stamp") or {}
            if isinstance(stamp, dict) and "sec" in stamp:
                self._sim_time = float(stamp["sec"]) + float(stamp.get("nsec", 0)) / 1e9
            self._health.joint_messages += 1
            self._health.joint_stream = True


class CameraReader:
    """Decode Gazebo camera frames into JPEG for the browser.

    Gazebo publishes raw ``gz.msgs.Image``. The browser cannot use that
    directly, so frames are re-encoded as JPEG here. The bound on frames kept is
    one: the viewer always shows the newest frame, so a slow client drops frames
    rather than falling behind the simulation.

    The encoder is Pillow, not OpenCV: ``cv2`` in this image fails to import
    against NumPy 2.5.3 (``numpy.core.multiarray failed to import``), and an
    optional dependency that cannot load is worse than not having it.
    """

    def __init__(self, topic: str, health: BridgeHealth, quality: int = 70) -> None:
        self._topic = topic
        self._health = health
        self._quality = quality
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._frame_index = 0
        self._stop = threading.Event()

    def latest(self) -> tuple[bytes | None, int]:
        """Return the newest (JPEG bytes, frame counter)."""
        with self._lock:
            return self._jpeg, self._frame_index

    def start(self) -> None:
        """Subscribe on a background thread."""
        threading.Thread(target=self._run, name="gz-camera-reader", daemon=True).start()

    def stop(self) -> None:
        """Unsubscribe from the camera topic."""
        self._stop.set()

    def _run(self) -> None:
        # Imported here so the module can be unit tested on a machine with no
        # Gazebo installed, which is most machines.
        try:
            from gz.msgs10.image_pb2 import Image
            from gz.transport13 import Node
            from PIL import Image as PILImage
        except ImportError as exc:  # pragma: no cover - no gz bindings
            self._health.errors.append(f"camera reader unavailable: {exc}")
            return

        node = Node()

        def on_image(msg: Any) -> None:
            if self._stop.is_set():
                return
            try:
                width, height = int(msg.width), int(msg.height)
                step = int(msg.step) or width * 3
                frame = PILImage.frombytes("RGB", (width, height), msg.data, "raw", "RGB", step)
                import io

                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG", quality=self._quality)
            except Exception as exc:
                self._health.errors.append(f"frame encode failed: {exc}")
                return
            with self._lock:
                self._jpeg = buffer.getvalue()
                self._frame_index += 1
                self._health.camera_frames += 1
                self._health.camera_stream = True

        node.subscribe(Image, self._topic, on_image)
        while not self._stop.is_set():
            time.sleep(0.2)


def robot_kinematics(urdf_path: str | Path) -> dict[str, Any]:
    """Extract the Panda's kinematic tree from its URDF, for browser rendering.

    The viewer needs joint origins, axes and limits to place links. Reading them
    out of the same URDF Gazebo loads means the rendered arm and the simulated
    arm cannot drift apart: there is one description of the robot, and this is a
    projection of it rather than a second, hand-maintained copy.

    Args:
        urdf_path: Path to the robot URDF.

    Returns:
        ``{"name": ..., "joints": [...], "links": [...]}``.
    """
    root = ElementTree.parse(Path(urdf_path)).getroot()
    joints: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        xyz = [
            float(v)
            for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
        ]
        rpy = [
            float(v)
            for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
        ]
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),  # type: ignore[union-attr]
                "child": joint.find("child").get("link"),  # type: ignore[union-attr]
                "xyz": xyz,
                "rpy": rpy,
                "axis": [
                    float(v)
                    for v in (axis.get("xyz", "0 0 1") if axis is not None else "0 0 1").split()
                ],
                "lower": float(limit.get("lower", -3.0)) if limit is not None else -3.0,
                "upper": float(limit.get("upper", 3.0)) if limit is not None else 3.0,
            }
        )
    return {
        "name": "panda",
        "joints": joints,
        "links": [link.get("name") for link in root.findall("link")],
    }


@dataclass
class DemoContext:
    """What the demo is running, shown in the viewer's status panel.

    ``config_hash`` is the value ROBOT CI already uses to identify a condition,
    so the browser reports the same identity an artifact would record.
    """

    task: str = "PickAndPlace"
    policy: str = "policy_v1"
    seed: int = 3
    depth_noise_mm: float = 0.0
    condition: str = "nominal"
    config_hash: str = ""
    phase: str = "idle"


class LiveBridge:
    """Serve the viewer and push live Gazebo state to it.

    The browser gets one JSON stream of joint state plus a binary camera stream.
    Both are read straight from the running simulation; if the simulation stops,
    the numbers stop changing, which is what makes the display trustworthy.
    """

    def __init__(
        self,
        *,
        dashboard_dir: Path,
        urdf_path: Path,
        joint_topic: str,
        camera_topic: str,
        context: DemoContext,
        port: int = 8000,
        scene: dict[str, Any] | None = None,
    ) -> None:
        self.health = BridgeHealth()
        self._port = port
        self._dashboard_dir = dashboard_dir
        self._context = context
        self._scene = scene or {}
        self._joints = JointStateReader(joint_topic, self.health)
        self._camera = CameraReader(camera_topic, self.health)
        self._kinematics = robot_kinematics(urdf_path)
        self.health.panda_loaded = bool(self._kinematics["joints"])

    def health_report(self) -> dict[str, Any]:
        """Return the current health report."""
        return self.health.as_dict()

    def _state_message(self) -> str:
        positions, sim_time = self._joints.latest()
        return json.dumps(
            {
                "type": "state",
                "sim_time": sim_time,
                "joints": positions,
                "context": {
                    "task": self._context.task,
                    "policy": self._context.policy,
                    "seed": self._context.seed,
                    "depth_noise_mm": self._context.depth_noise_mm,
                    "condition": self._context.condition,
                    "config_hash": self._context.config_hash,
                    "phase": self._context.phase,
                },
                "health": self.health.as_dict(),
            }
        )

    def run(self) -> None:
        """Start the readers, the static server and the WebSocket server."""
        import asyncio

        import websockets

        self._joints.start()
        self._camera.start()

        handler = _static_handler(self._dashboard_dir, self)
        server = ThreadingHTTPServer(("0.0.0.0", self._port), handler)
        threading.Thread(target=server.serve_forever, daemon=True, name="http").start()

        async def _stream(websocket: Any) -> None:
            while True:
                try:
                    await websocket.send(self._state_message())
                except websockets.exceptions.ConnectionClosed:
                    return
                await asyncio.sleep(0.05)

        async def _frames(websocket: Any) -> None:
            last = -1
            while True:
                jpeg, index = self._camera.latest()
                if jpeg is not None and index != last:
                    last = index
                    try:
                        await websocket.send(jpeg)
                    except websockets.exceptions.ConnectionClosed:
                        return
                await asyncio.sleep(0.05)

        async def _route(websocket: Any) -> None:
            # websockets >= 14 passes only the connection; the path lives on the
            # request. Reading it there keeps this working on the version the
            # image actually ships (17.1) rather than the older two-argument form.
            path = websocket.request.path
            if path == "/ws/state":
                await _stream(websocket)
            else:
                await _frames(websocket)

        async def _main() -> None:
            async with websockets.serve(
                _route,
                "0.0.0.0",
                self._port + 1,
                max_size=None,
            ):
                await asyncio.Future()

        asyncio.run(_main())


def _static_handler(directory: Path, bridge: LiveBridge) -> type[SimpleHTTPRequestHandler]:
    """Build the request handler that serves the viewer and the demo's JSON.

    Returning a class rather than a module-level handler keeps the bridge
    instance reachable from the request without a global.

    Besides the viewer this exposes the experiment launcher:

    * ``GET  /experiments.json`` -- the suite, the artifacts, and launcher status;
    * ``POST /experiments/start`` -- start one condition (refused if one runs);
    * ``GET  /artifacts/...``    -- the generated packages, for download.

    The launcher serialises runs deliberately: Gazebo's transport is global to
    the container, so two concurrent episodes would interleave on the same joint
    state and produce two results neither could be trusted.
    """
    from robot_ci.recording.launcher import ExperimentLauncher, artifacts_root

    launcher = ExperimentLauncher(root=artifacts_root())

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                payload = bridge.health_report()
                payload["ready"] = bridge.health.ready()
                self._json(payload)
                return
            if self.path == "/robot.json":
                self._json({**bridge._kinematics, "scene": bridge._scene})
                return
            if self.path == "/status.json":
                self._json(bridge.health_report())
                return
            if self.path == "/experiments.json":
                self._json(
                    {
                        "ready": bridge.health.ready(),
                        "status": launcher.status().to_dict(),
                        "suite": launcher.suite(),
                        "artifacts": _artifact_index(artifacts_root()),
                    }
                )
                return
            if self.path.startswith("/artifacts/"):
                self._artifact(self.path[len("/artifacts/"):])
                return
            if self.path == "/":
                bridge.health.viewer_served = True
            super().do_GET()

        def do_POST(self) -> None:
            if self.path != "/experiments/start":
                self._error(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._error(400, "invalid JSON body")
                return
            seed = body.get("seed")
            policy = body.get("policy")
            started, message = launcher.start(
                str(body.get("key", "")),
                seed=int(seed) if seed is not None else None,
                policy=str(policy) if policy else None,
            )
            if not started:
                self._error(409, message)
                return
            self._json(
                {"started": started, "message": message, "status": launcher.status().to_dict()}
            )

        def _artifact(self, relative: str) -> None:
            """Serve a file from the artifacts directory, refusing escapes."""
            root = artifacts_root().resolve()
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                self._error(404, "no such artifact")
                return
            payload = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _content_type(target))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            """Silence per-request logging; the demo logs its own milestones."""

    return Handler


def _artifact_index(root: Path) -> list[dict[str, Any]]:
    """List the artifacts currently on disk, so the dashboard can link them."""
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for package in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f.name for f in package.iterdir() if f.is_file())
        if not files:
            continue
        entries.append(
            {
                "key": package.name,
                "package": f"artifacts/{root.name}/{package.name}",
                "files": files,
                "zip": f"artifacts/{root.name}/{package.name}.zip"
                if (root / f"{package.name}.zip").exists()
                else None,
            }
        )
    for archive in sorted(root.glob("*.zip")):
        if (root / archive.stem).is_dir():
            continue  # already listed above with the package it belongs to
        entries.append({"key": archive.stem, "zip": f"artifacts/{root.name}/{archive.name}"})
    return entries


def _content_type(path: Path) -> str:
    """Map an artifact's extension to a content type."""
    return {
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".json": "application/json",
        ".md": "text/markdown; charset=utf-8",
        ".zip": "application/zip",
        ".txt": "text/plain; charset=utf-8",
    }.get(path.suffix, "application/octet-stream")


def main(argv: list[str] | None = None) -> int:
    """Run the bridge as a module: ``python -m robot_ci.gazebo.live``.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Live Gazebo -> browser bridge.")
    parser.add_argument("--dashboard", default="dashboard", type=Path)
    parser.add_argument("--urdf", default="gazebo/models/panda/panda.urdf", type=Path)
    parser.add_argument("--world", default=os.environ.get("ROBOT_CI_WORLD", "robot_ci_pick_place"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ROBOT_CI_PORT", "8000")))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--depth-noise-mm", type=float, default=0.0)
    parser.add_argument("--config-hash", default="")
    args = parser.parse_args(argv)

    # The table and object come from the same scenario functions that generate
    # the SDF, so the viewer draws where the simulation actually put them.
    from robot_ci.gazebo.world import object_pose_from_scenario
    from robot_ci.scenarios import default_scenario

    scenario = default_scenario()
    object_pose = object_pose_from_scenario(scenario)
    scene = {
        "table_height_m": float(scenario.environment.table_height_m),
        "workspace_center": [float(v) for v in scenario.environment.workspace_center],
        "object_pose": [float(v) for v in object_pose],
    }

    bridge = LiveBridge(
        dashboard_dir=args.dashboard,
        urdf_path=args.urdf,
        joint_topic=f"/world/{args.world}/model/panda/joint_state",
        camera_topic=(
            f"/world/{args.world}/model/camera_rig/link/link/sensor/robot_ci_camera/image"
        ),
        context=DemoContext(
            seed=args.seed,
            depth_noise_mm=args.depth_noise_mm,
            config_hash=args.config_hash,
            condition="nominal"
            if args.depth_noise_mm == 0
            else f"depth_noise_{args.depth_noise_mm:g}mm",
        ),
        port=args.port,
        scene=scene,
    )
    print(f"viewer on http://localhost:{args.port}", flush=True)
    bridge.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
