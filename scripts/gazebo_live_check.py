#!/usr/bin/env python3
"""Prove the live demo is really live, from the data the browser receives.

Run inside the demo container (or anywhere that can reach the viewer port).

This connects to the *same* WebSockets the browser uses and checks that the
joint values actually change over time and that camera frames decode as JPEG.
A bridge that merely accepted a connection would pass a weaker test; this one
fails unless Gazebo is genuinely moving the arm.

Usage:
    python3 scripts/gazebo_live_check.py [--port 8000] [--seconds 4]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request

__all__ = ["main"]


def _http(port: int, path: str) -> bytes:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return response.read()


def main() -> int:
    """Sample the live streams and report whether they carry real data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    from websockets.sync.client import connect

    failures: list[str] = []

    health = json.loads(_http(args.port, "/healthz"))
    print(f"healthz        ready={health.get('ready')} {health}")

    robot = json.loads(_http(args.port, "/robot.json"))
    joints = [joint["name"] for joint in robot["joints"]]
    expectations = [joint["name"] for joint in robot["joints"] if joint["type"] != "fixed"]
    print(f"robot.json     {len(joints)} joints: {', '.join(joints)}")
    print(f"               actuated: {', '.join(expectations)}")
    if len(joints) < 9:
        failures.append(f"robot.json exposes only {len(joints)} joints")
    if not robot.get("scene"):
        failures.append("robot.json carries no scene geometry")
    else:
        print(f"scene          table z={robot['scene']['table_height_m']} "
              f"object={[round(v, 4) for v in robot['scene']['object_pose']]}")

    # ---- joint stream: values must change, not merely arrive.
    samples: list[dict[str, float]] = []
    sim_times: list[float] = []
    with connect(f"ws://127.0.0.1:{args.port + 1}/ws/state") as socket:
        while len(samples) < 60:
            message = json.loads(socket.recv(timeout=5))
            samples.append(message["joints"])
            sim_times.append(message["sim_time"])

    print(f"state stream   {len(samples)} messages, sim time {sim_times[0]:.2f} -> {sim_times[-1]:.2f} s")
    if sim_times[-1] <= sim_times[0]:
        failures.append("simulation clock did not advance")

    print("joint travel over the sample window:")
    for name in joints:
        values = [sample.get(name, 0.0) for sample in samples]
        travel = max(values) - min(values)
        # hand_joint is the model's fixed wrist mount, not an actuator: it must
        # legitimately read zero for the entire run, which is itself proof the
        # stream is the real model and not a curve being swept everywhere.
        fixed = name not in expectations
        moved = "moving" if travel > 1e-4 else ("fixed" if fixed else "STATIC")
        print(f"    {name:<14} {min(values): .4f} .. {max(values): .4f}  travel {travel:.4f}  {moved}")
        if not fixed and travel <= 1e-4:
            failures.append(f"{name} never moved")
        if fixed and travel > 1e-4:
            failures.append(f"{name} should be fixed but moved")

    # ---- camera stream: frames must be real JPEGs *of real content*.
    #
    # Validating only the JPEG header would have passed while the camera was
    # aimed past the table, because a well-formed frame of nothing looks fine.
    # A frame that is one flat colour carries no information about the scene, so
    # it fails here.
    frames = 0
    sizes: list[int] = []
    last_payload: bytes | None = None
    with connect(f"ws://127.0.0.1:{args.port + 1}/ws/camera", max_size=None) as socket:
        while frames < 10:
            payload = socket.recv(timeout=10)
            if isinstance(payload, str):
                continue
            if not payload.startswith(b"\xff\xd8"):
                failures.append("camera frame is not a JPEG")
                break
            frames += 1
            sizes.append(len(payload))
            last_payload = payload
    if sizes:
        print(f"camera stream  {frames} JPEG frames, mean {statistics.mean(sizes):.0f} bytes")
    else:
        print("camera stream  no frames")
    if not frames or last_payload is None:
        failures.append("no camera frames received")
    else:
        try:
            from io import BytesIO

            from PIL import Image as PILImage

            image = PILImage.open(BytesIO(last_payload)).convert("RGB")
            colours = image.getcolors(maxcolors=image.width * image.height) or []
            print(
                f"              {image.width}x{image.height}, "
                f"{len(colours)} distinct colours"
            )
            if len(colours) < 8:
                failures.append(
                    f"camera frame is flat ({len(colours)} distinct colours): "
                    "the camera is seeing nothing, not the scene"
                )
        except ImportError:
            print("              Pillow unavailable; content not verified")

    print()
    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"    - {failure}")
        return 1
    print("PASS: live joint state, live camera and scene geometry all confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
