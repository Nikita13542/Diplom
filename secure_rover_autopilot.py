import ipaddress
import hashlib
import hmac
import json
import logging
import os
import random
import secrets
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


class CameraMode(str, Enum):
    IP_WEBCAM = "IP_WEBCAM"
    DROIDCAM = "DROIDCAM"
    USB = "USB"


class TargetColor(str, Enum):
    GREEN = "GREEN"
    RED = "RED"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"


@dataclass(frozen=True)
class SecurityConfig:
    allowed_rover_ip: str = os.getenv("ROVER_IP", "192.168.4.1")
    allowed_phone_ip: str = os.getenv("PHONE_IP", "192.168.4.2")
    request_timeout_s: float = 0.15
    send_delay_s: float = 0.1
    command_log_interval_s: float = 1.0
    max_speed: float = 0.5
    max_failed_commands: int = 5
    fail_safe_stop_s: float = 0.5
    hmac_secret: str = os.getenv("ROVER_HMAC_SECRET", "")
    log_file: Path = Path(os.getenv("ROVER_SECURITY_LOG", "rover_security.log"))


@dataclass(frozen=True)
class RoverConfig:
    camera_mode: CameraMode = CameraMode(os.getenv("CAMERA_MODE", "IP_WEBCAM"))
    target_color: TargetColor = TargetColor(os.getenv("TARGET_COLOR", "GREEN"))
    process_fps: float = float(os.getenv("PROCESS_FPS", "10.0"))
    base_speed: float = 0.18
    track_speed: float = 0.20
    target_min_area: int = 600
    target_close_area: int = 4500
    frame_width: int = 640
    frame_height: int = 480
    startup_observe_s: float = 3.0
    startup_scan_s: float = 5.5
    startup_scan_speed: float = 0.23
    kp: float = 0.24
    ki: float = 0.015
    kd: float = 0.09
    max_integral: float = 1.0
    max_steer: float = 0.16
    center_deadband: float = 0.045
    target_center_smoothing: float = 0.55
    lost_target_limit: int = 8
    stuck_seconds: float = 5.0
    stuck_frame_diff_threshold: float = 2.2
    stuck_reverse_s: float = 0.7
    stuck_turn_s: float = 0.9
    stuck_reverse_speed: float = -0.28
    stuck_turn_speed: float = 0.28
    safe_edge_density: float = 0.32
    critical_edge_density: float = 0.58
    smoothing: float = 0.25


SECURITY = SecurityConfig()
CONFIG = RoverConfig()


def setup_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("secure_rover")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


LOGGER = setup_logging(SECURITY.log_file)
cv2.setUseOptimized(True)


@dataclass
class FrameAnalysis:
    frame: np.ndarray
    motion_signature: np.ndarray
    bottom_edges: np.ndarray
    edge_density: float
    target_found: bool
    target_area: float
    target_cx: int
    target_bgr: Tuple[int, int, int]


def validate_private_ip(value: str, expected: str, label: str) -> str:
    try:
        ip = ipaddress.ip_address(value)
        expected_ip = ipaddress.ip_address(expected)
    except ValueError as exc:
        raise ValueError(f"{label}: invalid IP address") from exc

    if ip != expected_ip:
        raise ValueError(f"{label}: IP {ip} is not in the allowlist")
    if not ip.is_private:
        raise ValueError(f"{label}: only private lab-network IPs are allowed")
    return str(ip)


def build_url(ip: str, port: Optional[int], path: str) -> str:
    host = validate_private_ip(ip, ip, "host")
    if port is None:
        url = f"http://{host}{path}"
    else:
        if not 1 <= port <= 65535:
            raise ValueError("port is out of range")
        url = f"http://{host}:{port}{path}"

    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != host:
        raise ValueError("unsafe URL was rejected")
    return url


def validate_speed(value: float, max_speed: float) -> float:
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("motor speed must be a finite number")
    return max(-max_speed, min(max_speed, float(value)))


def frame_interval(config: RoverConfig) -> float:
    if config.process_fps <= 0:
        raise ValueError("process_fps must be greater than zero")
    return 1.0 / config.process_fps


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign_payload(payload: dict, secret: str) -> str:
    message = canonical_json(payload).encode("utf-8")
    key = secret.encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_signed_payload(payload: dict, secret: str) -> bool:
    received_signature = payload.get("sig")
    if not isinstance(received_signature, str):
        return False

    unsigned_payload = {key: value for key, value in payload.items() if key != "sig"}
    expected_signature = sign_payload(unsigned_payload, secret)
    return hmac.compare_digest(received_signature, expected_signature)


class RoverController:
    def __init__(self, security: SecurityConfig = SECURITY):
        rover_ip = validate_private_ip(
            security.allowed_rover_ip,
            security.allowed_rover_ip,
            "rover",
        )
        self.api_url = build_url(rover_ip, None, "/js")
        self.security = security
        self.last_send = 0.0
        self.last_left = 0.0
        self.last_right = 0.0
        self.last_command_log = 0.0
        self.failed_commands = 0
        self.sequence = 0
        self.session = requests.Session()
        self.session.trust_env = False  # ignore proxy env vars to reduce command exfiltration risk

        LOGGER.info("controller_initialized api_url=%s", self.api_url)
        if self.security.hmac_secret:
            LOGGER.info("hmac_signing_enabled algorithm=HMAC-SHA256")
        else:
            LOGGER.warning("hmac_signing_disabled set ROVER_HMAC_SECRET to sign commands")

    def build_motor_payload(self, left: float, right: float) -> dict:
        self.sequence += 1
        payload = {"T": 1, "L": round(left, 3), "R": round(right, 3)}
        if not self.security.hmac_secret:
            return payload

        signed_fields = {
            **payload,
            "ts": int(time.time()),
            "nonce": secrets.token_hex(8),
            "seq": self.sequence,
        }
        signed_fields["sig"] = sign_payload(signed_fields, self.security.hmac_secret)
        return signed_fields

    def send_motor(self, left: float, right: float) -> bool:
        left = validate_speed(left, self.security.max_speed)
        right = validate_speed(right, self.security.max_speed)

        left = CONFIG.smoothing * self.last_left + (1 - CONFIG.smoothing) * left
        right = CONFIG.smoothing * self.last_right + (1 - CONFIG.smoothing) * right
        left = validate_speed(left, self.security.max_speed)
        right = validate_speed(right, self.security.max_speed)

        now = time.monotonic()
        if now - self.last_send <= self.security.send_delay_s:
            self.last_left = left
            self.last_right = right
            return True

        payload = self.build_motor_payload(left, right)
        try:
            response = self.session.get(
                self.api_url,
                params={"json": canonical_json(payload)},
                timeout=self.security.request_timeout_s,
                allow_redirects=False,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            self.failed_commands += 1
            LOGGER.warning(
                "command_failed failures=%s payload=%s error=%s",
                self.failed_commands,
                payload,
                exc.__class__.__name__,
            )
            if self.failed_commands >= self.security.max_failed_commands:
                self.emergency_stop("too_many_command_failures")
            return False

        self.failed_commands = 0
        self.last_send = now
        self.last_left = left
        self.last_right = right
        if now - self.last_command_log >= self.security.command_log_interval_s:
            LOGGER.info("command_sent left=%.3f right=%.3f", left, right)
            self.last_command_log = now
        return True

    def emergency_stop(self, reason: str) -> None:
        LOGGER.error("emergency_stop reason=%s", reason)
        self.last_left = 0.0
        self.last_right = 0.0
        for _ in range(3):
            try:
                payload = self.build_motor_payload(0, 0)
                self.session.get(
                    self.api_url,
                    params={"json": canonical_json(payload)},
                    timeout=self.security.request_timeout_s,
                    allow_redirects=False,
                )
            except requests.RequestException:
                pass
            time.sleep(self.security.fail_safe_stop_s / 3)

    def stop(self) -> None:
        self.send_motor(0, 0)


class LatestFrameCamera:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_ok = False
        self.running = True
        self.thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.latest_ok = ok
                if ok:
                    self.latest_frame = frame
            if not ok:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.latest_ok or self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


CameraSource = Union[cv2.VideoCapture, LatestFrameCamera]


def open_camera(security: SecurityConfig, config: RoverConfig) -> LatestFrameCamera:
    phone_ip = validate_private_ip(
        security.allowed_phone_ip,
        security.allowed_phone_ip,
        "camera",
    )
    if config.camera_mode == CameraMode.IP_WEBCAM:
        source = build_url(phone_ip, 8080, "/video")
    elif config.camera_mode == CameraMode.DROIDCAM:
        source = build_url(phone_ip, 4747, "/video")
    else:
        source = 1

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("camera_open_failed")
    LOGGER.info("camera_opened mode=%s", config.camera_mode.value)
    return LatestFrameCamera(cap)


def color_mask(hsv: np.ndarray, target_color: TargetColor) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    if target_color == TargetColor.GREEN:
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        return mask, (0, 255, 0)
    if target_color == TargetColor.RED:
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        return cv2.bitwise_or(mask1, mask2), (0, 0, 255)
    if target_color == TargetColor.ORANGE:
        mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
        return mask, (0, 165, 255)

    mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    return mask, (0, 255, 255)


def analyze_frame(frame: np.ndarray, config: RoverConfig) -> FrameAnalysis:
    frame = cv2.resize(frame, (config.frame_width, config.frame_height))
    h, w = frame.shape[:2]
    cx = w // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_signature = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    bottom_edges = edges[int(h * 0.65):, :]
    edge_density = float(np.mean(bottom_edges > 0))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask, target_bgr = color_mask(hsv, config.target_color)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_found = False
    target_area = 0.0
    target_cx = cx

    if cnts:
        contour = max(cnts, key=cv2.contourArea)
        target_area = float(cv2.contourArea(contour))
        if target_area > config.target_min_area:
            moments = cv2.moments(contour)
            x, y, ww, hh = cv2.boundingRect(contour)
            if moments["m00"] > 0:
                target_cx = int(moments["m10"] / moments["m00"])
            else:
                target_cx = x + ww // 2
            target_found = True
            cv2.rectangle(frame, (x, y), (x + ww, y + hh), target_bgr, 3)
            cv2.circle(frame, (target_cx, y + hh // 2), 6, target_bgr, -1)

    return FrameAnalysis(
        frame=frame,
        motion_signature=motion_signature,
        bottom_edges=bottom_edges,
        edge_density=edge_density,
        target_found=target_found,
        target_area=target_area,
        target_cx=target_cx,
        target_bgr=target_bgr,
    )


def show_status(frame: np.ndarray, mode: str, analysis: FrameAnalysis, error: float = 0.0) -> None:
    cv2.putText(frame, f"MODE: {mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    if analysis.target_found:
        cv2.putText(
            frame,
            f"Area: {int(analysis.target_area)} Err: {error:+.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            analysis.target_bgr,
            2,
        )
    cv2.line(frame, (CONFIG.frame_width // 2, 0), (CONFIG.frame_width // 2, CONFIG.frame_height), (255, 255, 255), 1)
    cv2.imshow("Secure Rover Autopilot", frame)


def recover_from_stuck(controller: RoverController, reason: str, config: RoverConfig) -> None:
    LOGGER.warning("stuck_recovery_started reason=%s", reason)
    controller.emergency_stop(reason)
    controller.send_motor(config.stuck_reverse_speed, config.stuck_reverse_speed)
    time.sleep(config.stuck_reverse_s)
    turn = random.SystemRandom().choice([-config.stuck_turn_speed, config.stuck_turn_speed])
    controller.send_motor(turn, -turn)
    time.sleep(config.stuck_turn_s)
    controller.emergency_stop("stuck_recovery_complete")


def frame_difference(prev: Optional[np.ndarray], current: np.ndarray) -> float:
    if prev is None:
        return 255.0
    diff = cv2.absdiff(prev, current)
    return float(np.mean(diff))


def startup_environment_scan(controller: RoverController, cap: CameraSource, config: RoverConfig) -> bool:
    LOGGER.info("startup_observe_started seconds=%.1f", config.startup_observe_s)
    controller.emergency_stop("startup_observe")
    interval = frame_interval(config)
    next_process_at = 0.0
    deadline = time.monotonic() + config.startup_observe_s
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_process_at:
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt
            time.sleep(min(0.01, next_process_at - now))
            continue

        ok, frame = cap.read()
        next_process_at = time.monotonic() + interval
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        analysis = analyze_frame(frame, config)
        show_status(analysis.frame, "OBSERVE", analysis)
        if analysis.target_found:
            controller.emergency_stop("startup_target_detected")
            LOGGER.info("startup_observe_target_detected area=%.1f", analysis.target_area)
            return True
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    LOGGER.info("startup_scan_started seconds=%.1f speed=%.2f", config.startup_scan_s, config.startup_scan_speed)
    deadline = time.monotonic() + config.startup_scan_s
    best_area = 0.0
    next_process_at = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_process_at:
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt
            time.sleep(min(0.01, next_process_at - now))
            continue

        controller.send_motor(config.startup_scan_speed, -config.startup_scan_speed)
        ok, frame = cap.read()
        next_process_at = time.monotonic() + interval
        if ok and frame is not None:
            analysis = analyze_frame(frame, config)
            if analysis.target_found and analysis.target_area > best_area:
                best_area = analysis.target_area
                controller.emergency_stop("startup_scan_target_detected")
                LOGGER.info("startup_scan_target_detected area=%.1f", best_area)
                show_status(analysis.frame, "TARGET", analysis)
                return True
            show_status(analysis.frame, "SCAN360", analysis)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    controller.emergency_stop("startup_scan_complete")
    LOGGER.info("startup_scan_complete best_target_area=%.1f", best_area)
    return False


def run_autopilot() -> None:
    controller = RoverController()
    cap = open_camera(SECURITY, CONFIG)

    last_error = 0.0
    integral = 0.0
    mode = "SEARCH"
    action_done = False
    filtered_target_cx: Optional[float] = None
    lost_target_frames = 0
    previous_motion_signature: Optional[np.ndarray] = None
    stuck_since: Optional[float] = None
    interval = frame_interval(CONFIG)
    next_process_at = 0.0

    try:
        LOGGER.info("autopilot_started")
        startup_found_target = startup_environment_scan(controller, cap, CONFIG)
        LOGGER.info("startup_finished target_found=%s", startup_found_target)
        while True:
            now = time.monotonic()
            if now < next_process_at:
                if cv2.waitKey(1) & 0xFF == 27:
                    LOGGER.info("operator_exit")
                    break
                time.sleep(min(0.01, next_process_at - now))
                continue

            ok, frame = cap.read()
            next_process_at = time.monotonic() + interval
            if not ok or frame is None:
                LOGGER.warning("camera_frame_missing")
                controller.emergency_stop("camera_frame_missing")
                time.sleep(0.05)
                continue

            analysis = analyze_frame(frame, CONFIG)
            frame = analysis.frame
            _, w = frame.shape[:2]
            cx = w // 2
            now = time.monotonic()

            diff = frame_difference(previous_motion_signature, analysis.motion_signature)
            previous_motion_signature = analysis.motion_signature.copy()
            if diff < CONFIG.stuck_frame_diff_threshold:
                if stuck_since is None:
                    stuck_since = now
                elif now - stuck_since >= CONFIG.stuck_seconds:
                    show_status(frame, "STUCK", analysis)
                    recover_from_stuck(controller, "static_frame_watchdog", CONFIG)
                    previous_motion_signature = None
                    stuck_since = None
                    filtered_target_cx = None
                    integral = 0.0
                    last_error = 0.0
                    lost_target_frames = CONFIG.lost_target_limit + 1
                    next_process_at = time.monotonic() + interval
                    continue
            else:
                stuck_since = None

            if analysis.target_found:
                lost_target_frames = 0
                if filtered_target_cx is None:
                    filtered_target_cx = float(analysis.target_cx)
                else:
                    alpha = CONFIG.target_center_smoothing
                    filtered_target_cx = alpha * filtered_target_cx + (1 - alpha) * analysis.target_cx
            else:
                lost_target_frames += 1

            target_full_area = 0.78 * CONFIG.frame_width * CONFIG.frame_height
            if analysis.target_found and analysis.target_area > target_full_area:
                mode = "ACTION"
            elif analysis.target_found or lost_target_frames <= CONFIG.lost_target_limit:
                mode = "TRACK"
            else:
                mode = "SEARCH"

            left = right = 0.0
            normalized_error = 0.0
            if mode == "ACTION" and not action_done:
                controller.emergency_stop("target_reached")
                LOGGER.info("target_reached")
                for _ in range(4):
                    controller.send_motor(0.4, -0.4)
                    time.sleep(0.35)
                    controller.send_motor(-0.4, 0.4)
                    time.sleep(0.35)
                action_done = True
                controller.emergency_stop("mission_complete")
                break

            elif mode == "TRACK":
                if filtered_target_cx is None:
                    filtered_target_cx = float(cx)

                normalized_error = (filtered_target_cx - cx) / (w / 2)
                normalized_error = max(-1.0, min(1.0, normalized_error))
                if abs(normalized_error) < CONFIG.center_deadband:
                    normalized_error = 0.0

                integral += normalized_error
                integral = max(-CONFIG.max_integral, min(CONFIG.max_integral, integral))
                derivative = normalized_error - last_error
                steer = CONFIG.kp * normalized_error + CONFIG.ki * integral + CONFIG.kd * derivative
                last_error = normalized_error
                steer = max(-CONFIG.max_steer, min(CONFIG.max_steer, steer))

                base_speed = CONFIG.track_speed
                if analysis.target_area > CONFIG.target_close_area:
                    base_speed *= 0.6
                if lost_target_frames:
                    base_speed *= 0.45
                    steer = max(-CONFIG.max_steer, min(CONFIG.max_steer, steer * 1.35))

                left = base_speed + steer
                right = base_speed - steer

            elif mode == "SEARCH":
                integral = 0.0
                filtered_target_cx = None
                if analysis.edge_density > CONFIG.critical_edge_density:
                    controller.emergency_stop("critical_obstacle")
                    controller.send_motor(-0.35, -0.35)
                    time.sleep(0.5)
                    turn_dir = random.SystemRandom().choice([-0.3, 0.3])
                    controller.send_motor(turn_dir, -turn_dir)
                    time.sleep(0.8)
                elif analysis.edge_density > CONFIG.safe_edge_density:
                    left_half = float(np.mean(analysis.bottom_edges[:, :w // 2]))
                    right_half = float(np.mean(analysis.bottom_edges[:, w // 2:]))
                    left = CONFIG.base_speed * 0.6
                    right = CONFIG.base_speed * (1.4 if left_half > right_half else 0.6)
                else:
                    left = right = CONFIG.base_speed

            controller.send_motor(left, right)
            show_status(frame, mode, analysis, normalized_error)

            if cv2.waitKey(1) & 0xFF == 27:
                LOGGER.info("operator_exit")
                break
    finally:
        controller.emergency_stop("shutdown")
        cap.release()
        cv2.destroyAllWindows()
        LOGGER.info("autopilot_stopped")


if __name__ == "__main__":
    run_autopilot()
