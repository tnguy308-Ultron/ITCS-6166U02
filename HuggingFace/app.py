"""HuggingFace Space (Gradio + ZeroGPU) for GVHMR — World-Grounded Human Motion Recovery.

Upload a video → recover SMPL human motion → get a side-by-side (in-camera + world)
mesh overlay video plus an .npz of the SMPL parameters.

Uses Hugging Face **ZeroGPU** (free shared GPU for visitors; the Space owner selects the
ZeroGPU hardware flavor in Settings — requires a PRO/Team/Enterprise plan to host). Only the GPU
work (preproc models + the network + output serialization) runs inside ``@spaces.GPU``
(``_recover_gpu``); dependency bootstrap, preflight, and moderngl rendering run OUTSIDE it, so the short ZeroGPU
budget isn't spent on non-CUDA work.

Body models: configure a private mirror (`GVHMR_BODY_MODELS_MIRROR` + `HF_TOKEN`) or MPI
Secrets — see space/README.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

# ZeroGPU: `spaces` must be imported before any torch/CUDA import (HF requirement). gradio above is fine —
# it doesn't import torch — and torch is only imported later, well after this line.
import spaces


def _patch_torch_dynamo_config() -> None:
    """ZeroGPU's torch 2.11.0+cu130 has an internal skew: ``torch._dynamo/config.py`` passes
    ``Config(deprecated=…)`` but ``torch.utils._config_module.Config`` doesn't declare it, so
    ``import torch._dynamo`` (hence ``import pytorch_lightning``) dies with ``TypeError: Config() got an
    unexpected keyword argument 'deprecated'``. Wrap Config.__init__ to drop kwargs it doesn't accept —
    run BEFORE anything imports torch._dynamo. Dropping deprecation metadata is harmless (we never
    torch.compile); a torch build that already accepts the kwarg is left untouched."""
    try:
        import functools
        import inspect

        import torch.utils._config_module as cm

        accepted = set(inspect.signature(cm.Config).parameters)
        if "deprecated" in accepted:
            return  # already-fixed torch — nothing to do
        _orig = cm.Config  # a factory FUNCTION (not a class), so wrap the callable

        @functools.wraps(_orig)
        def _wrapped(*args, **kwargs):
            return _orig(*args, **{k: v for k, v in kwargs.items() if k in accepted})

        cm.Config = _wrapped
    except Exception as exc:  # noqa: BLE001 — diagnostic only; never block startup on the shim
        print(f"[gvhmr] torch _dynamo Config shim skipped ({type(exc).__name__}: {exc})")


_patch_torch_dynamo_config()

# The moderngl overlay renderer needs a GL context; on a GPU-less Space (cpu-basic) there's no hardware
# GL ("libEGL.so not loaded"), so force Mesa's software rasterizer (llvmpipe) — with the EGL libs from
# packages.txt this gives a headless context so rendering works instead of failing to pytorch3d (absent).
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


def _patch_gradio_file_whitelist() -> None:
    """SSR mode re-fetches uploads over HTTP; HF xethub bridge hosts must be allow-listed."""
    try:
        from gradio import processing_utils as pu

        for host in (
            "cas-bridge-direct.xethub.hf.co",
            "cas-bridge.xethub.hf.co",
            "xethub.hf.co",
        ):
            if host not in pu.PUBLIC_HOSTNAME_WHITELIST:
                pu.PUBLIC_HOSTNAME_WHITELIST.append(host)
    except Exception:  # noqa: BLE001 — gradio internals differ across versions
        pass


_patch_gradio_file_whitelist()

REPO_URL = "https://github.com/ryanrudes/gvhmr"

DEFAULT_HUB_REPO = os.getenv("GVHMR_HUB_REPO", "ryanrudes/gvhmr")

# All presets from gvhmr/configs/{detector,pose2d,backbone,camera}/ — keep in sync with the repo.
DETECTOR_CHOICES = [
    "yolo",
    "yolo11l",
    "yolo11m",
    "yolo11n",
    "yolo11s",
    "yolo11x",
    "yolo12l",
    "yolo12m",
    "yolo12n",
    "yolo12s",
    "yolo12x",
    "yolo26l",
    "yolo26m",
    "yolo26n",
    "yolo26s",
    "yolo26x",
    "yolov10b",
    "yolov10l",
    "yolov10m",
    "yolov10n",
    "yolov10s",
    "yolov10x",
    "yolov8l",
    "yolov8m",
    "yolov8n",
    "yolov8s",
    "yolov8x",
    "yolov9c",
    "yolov9e",
    "yolov9m",
    "yolov9s",
    "yolov9t",
]
POSE2D_CHOICES = ["vitpose", "rtmpose"]  # rtmpose: lazy-pip rtmlib+onnxruntime on first use
BACKBONE_CHOICES = ["hmr2"]  # dinov2 needs a retrained checkpoint
CAMERA_CHOICES = ["simplevo"]  # dpvo/dust3r/vggt are not set up on this Space

name = "Team 5"


DESCRIPTION = f"""
# GVHMR — World-Grounded Human Motion Recovery 
## Team 5 - Tom Nguyen, Brody Ehorn

Upload a video of a person and GVHMR recovers their **SMPL** body motion in both the
**camera** frame and a gravity-aligned **world** frame. You get a side-by-side mesh
overlay video and an `.npz` of the SMPL parameters.

*World-Grounded Human Motion Recovery via Gravity-View Coordinates* (SIGGRAPH Asia 2024).
Source & docs: [{REPO_URL}]({REPO_URL}). 

 > **ZeroGPU** (Space owner selects that hardware) is much faster; **CPU basic** also works but
 > is slow — keep clips short (~10–15 s), use *Static camera*, and skip flip-test when possible.
 > On ZeroGPU, each inference call gets **60 seconds of GPU time** on the free tier (HF default for
 > `@spaces.GPU`); PRO owners can raise the cap with a `ZERO_GPU_DURATION` Secret (seconds). 
"""

def _hub_repo_choices() -> list[str]:
    choices = [DEFAULT_HUB_REPO, "camenduru/GVHMR"]
    if extra := os.getenv("GVHMR_HUB_REPO_OPTIONS", ""):
        choices.extend(item.strip() for item in extra.split(",") if item.strip())
    return list(dict.fromkeys(choices))


def _apply_bootstrap_smpl_compat() -> None:
    """Legacy shims before ``gvhmr`` is installed (mirrors ``gvhmr.utils._smpl_compat``)."""
    import inspect

    import numpy as np

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, target in {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "unicode": np.str_,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, target)


def _pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def _bootstrap_deps() -> None:
    """Install chumpy (no-build-isolation) then gvhmr — cannot run at Space build time."""
    _apply_bootstrap_smpl_compat()

    try:
        import chumpy  # noqa: F401
    except ImportError:
        _pip_install("--no-build-isolation", "numpy>=1.26", "chumpy==0.70")

    try:
        import gvhmr  # noqa: F401
    except ImportError:
        # ZeroGPU preinstalls torch 2.11 — chumpy must be present first (see requirements.txt).
        _pip_install("gvhmr[preproc]>=1.2.0")

    _verify_preproc_imports()

    from gvhmr.utils._smpl_compat import apply

    apply()


def _verify_preproc_imports() -> None:
    """Fail fast if the runtime preproc extra did not land (common on first pip bootstrap)."""
    import importlib

    missing: list[str] = []
    for mod in ("ultralytics", "pycolmap", "cv2"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        return
    _pip_install("gvhmr[preproc]>=1.2.0", "pycolmap>=0.6", "opencv-python-headless>=4.8")
    still: list[str] = []
    for mod in missing:
        try:
            importlib.import_module(mod)
        except ImportError:
            still.append(mod)
    if still:
        raise RuntimeError(f"preproc dependencies missing after bootstrap: {', '.join(still)}")


def _ensure_pose2d_deps(pose2d: str) -> None:
    """Lazy-install RTMPose extras (rtmlib + onnxruntime) when selected — keeps vitpose-only runs lighter."""
    if pose2d != "rtmpose":
        return
    import importlib

    try:
        importlib.import_module("rtmlib")
    except ImportError:
        _pip_install("rtmlib", "onnxruntime>=1.16")
        try:
            importlib.import_module("rtmlib")
        except ImportError as exc:
            raise RuntimeError("RTMPose (rtmlib) failed to install — try again or use vitpose.") from exc


_BOOTSTRAPPED = False


def _ensure_bootstrapped() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _bootstrap_deps()
    _BOOTSTRAPPED = True


def _startup_bootstrap() -> None:
    """Best-effort dependency install at app *startup* (module import), not inside ``@spaces.GPU``.

    ZeroGPU reservations are short (60 s on the free tier); running the multi-minute chumpy/gvhmr[preproc]
    install here — once, before any request is GPU-scheduled — keeps that budget for the actual inference.
    ``spaces`` is already imported (top of file), so importing torch during the install is allowed. On
    failure we leave ``_BOOTSTRAPPED`` False so ``run`` retries in-request as a fallback.
    """
    try:
        _ensure_bootstrapped()
    except Exception:  # noqa: BLE001 — don't crash module import; run() will retry and surface a gr.Error
        traceback.print_exc()


# --------------------------------------------------------------------------------------
# Pipeline cache: lazy load inside @spaces.GPU (keyed by hub repo + stage presets).
# --------------------------------------------------------------------------------------
_PIPES: dict[tuple, Any] = {}


def _resolve_device() -> str:
    """Pick cuda → mps → cpu; honour ``GVHMR_DEVICE`` but fall back when unavailable.

    Uses only ``torch`` (HF preinstall) so this works before the runtime ``gvhmr`` pip bootstrap.
    """
    import torch

    def _auto() -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            return "mps"
        return "cpu"

    prefer = os.getenv("GVHMR_DEVICE", "").strip()
    if prefer == "cuda" and not torch.cuda.is_available():
        return _auto()
    if prefer == "mps":
        mps = getattr(torch.backends, "mps", None)
        if not (mps and mps.is_available()):
            return _auto()
    if prefer:
        return prefer
    return _auto()


def _cpu_preproc_overrides(pose2d: str, backbone: str) -> list[str]:
    """On a CPU Space, shrink the ViT batch so activations don't OOM the 16 GB cpu-basic box: batch 32 fp32
    ViT-Huge (ViTPose + HMR2) is fine on a GPU but tips CPU RAM over. No-op on a GPU Space.

    Guarded against BOTH failure modes: (1) the Space pins gvhmr from PyPI, and a release predating the
    batch_size knob would ``ConfigCompositionException``; (2) a stage the user SELECTED (e.g. rtmpose) may
    have no batch_size key even when the default vitpose does. So compose the demo cfg with the ACTUAL
    selected pose2d/backbone and only emit an override for a key that resolves on that composed config.
    """
    if _resolve_device() != "cpu":
        return []
    try:
        from hydra import compose, initialize_config_module
        from omegaconf import OmegaConf

        from gvhmr.configs import register_store_gvhmr

        register_store_gvhmr()
        sel = [f"pose2d={pose2d}", f"backbone={backbone}", "video_name=_"]
        with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
            cfg = compose(config_name="demo", overrides=sel)
        ov = []
        if OmegaConf.select(cfg, "pose2d.batch_size") is not None:
            ov.append("pose2d.batch_size=4")
        if OmegaConf.select(cfg, "backbone.batch_size") is not None:
            ov.append("backbone.batch_size=4")
        return ov
    except Exception:  # noqa: BLE001 — never let a memory-tuning override break inference
        return []


def _pipeline_key(
    model_repo: str,
    revision: str,
    detector: str,
    pose2d: str,
    backbone: str,
) -> tuple:
    rev = (revision or "").strip()
    return (model_repo.strip(), rev, detector, pose2d, backbone, _resolve_device())


def _get_pipeline(
    *,
    model_repo: str,
    revision: str,
    detector: str,
    pose2d: str,
    backbone: str,
):
    """Return a cached pipeline for the selected weights + preprocessing presets."""
    key = _pipeline_key(model_repo, revision, detector, pose2d, backbone)
    if key in _PIPES:
        return _PIPES[key]
    try:
        _ensure_bootstrapped()
        import gvhmr

        rev = (revision or "").strip() or None
        pipe = gvhmr.pipeline(
            "human-motion-recovery",
            model=model_repo.strip(),
            revision=rev,
            detector=detector,
            pose2d=pose2d,
            backbone=backbone,
            device=key[-1],
        )
        _PIPES[key] = pipe
        return pipe
    except Exception as exc:  # noqa: BLE001 — surface to the UI but do NOT cache the failure:
        # a transient first-call error (cold-bootstrap timeout, network blip, creds not yet set) must not
        # permanently disable this combo — the next request retries from scratch.
        traceback.print_exc()
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def _preflight_stages(*, static_camera: bool, camera: str, detector: str, pose2d: str, backbone: str) -> None:
    """Mirror ``ensure_deps`` checks but raise ``gr.Error`` (never ``SystemExit``)."""
    from gvhmr.utils.preproc.base import missing_requirements

    if pose2d not in POSE2D_CHOICES:
        raise gr.Error(f"2D pose '{pose2d}' is not available on this Space. Use vitpose or rtmpose.")
    _ensure_pose2d_deps(pose2d)
    if backbone not in BACKBONE_CHOICES:
        raise gr.Error(f"Backbone '{backbone}' is not available on this Space. Use hmr2.")
    if not static_camera and camera not in CAMERA_CHOICES:
        raise gr.Error(f"Camera '{camera}' is not set up on this Space. Check Static camera, or use simplevo.")

    selections: dict[str, str] = {"detector": detector, "pose2d": pose2d, "backbone": backbone}
    if not static_camera:
        selections["camera"] = camera
    missing = missing_requirements(selections)
    if missing:
        detail = "; ".join(f"{why} requires {mod}" for why, mod, _ in missing)
        raise gr.Error(f"Missing dependencies for this run: {detail}")


def _is_credentials_error(exc: Exception) -> bool:
    """Heuristic: does this exception look like missing gated-body-model credentials?"""
    text = f"{type(exc).__name__} {exc}".lower()
    # Specific to the gated body-model flow. Deliberately NOT bare "401"/"403"/"mirror"/"permissionerror":
    # a generic Hub 403 (rate limit, private checkpoint repo) is not a body-model creds problem, and
    # mislabelling it "SMPL/SMPL-X are gated…" sends the user down the wrong fix.
    needles = (
        "smplx_user",
        "smplx_pw",
        "smpl_user",
        "smpl_pw",
        "body model",
        "body_model",
        "registration-gated",
        "smpl-x body model",
        "is.tue.mpg.de",
    )
    return any(n in text for n in needles)


def _seconds_of_motion(num_frames: int, fps: float) -> float:
    return round(num_frames / fps, 2) if fps else 0.0


def _video_path(video) -> str | None:
    """Normalize Gradio Video value (path str or FileData dict) to a filesystem path."""
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        return video.get("path") or video.get("video")
    path = getattr(video, "path", None)
    return str(path) if path else str(video)


def _intrinsics_path(f) -> str | None:
    """Normalize an optional Gradio File value (path str / FileData dict / temp obj) to a path, or None.

    The intrinsics upload is optional — an empty field returns None and the pipeline falls back to video
    metadata / the FOV heuristic (see docs/CAMERA_METADATA.md)."""
    if not f:
        return None
    if isinstance(f, str):
        return f or None
    if isinstance(f, dict):
        return f.get("path") or f.get("name")
    path = getattr(f, "path", None) or getattr(f, "name", None)
    return str(path) if path else None


def _zero_gpu_duration_cap() -> int:
    """Max seconds requested per ``@spaces.GPU`` call (HF free tier default: 60)."""
    return int(os.getenv("ZERO_GPU_DURATION", "60"))


def _estimate_gpu_duration(
    video_path,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
    *args,
    **kwargs,
) -> int:
    """ZeroGPU queue budget (seconds) for one ``@spaces.GPU`` call — clamped to the per-call cap.

    HF's free tier caps every call at 60 s; PRO owners can raise it via ``$ZERO_GPU_DURATION``. Deps are
    bootstrapped at app startup (see ``_startup_bootstrap``), so the GPU reservation only pays for the
    checkpoint load + inference, not pip.
    """
    cap = _zero_gpu_duration_cap()

    video_path = _video_path(video_path)
    secs = 30.0
    if video_path:
        try:
            import cv2

            cap_vid = cv2.VideoCapture(video_path)
            fps = cap_vid.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap_vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap_vid.release()
            if fps > 0 and frames > 0:
                secs = float(frames / fps)
        except Exception:  # noqa: BLE001
            pass
    per_sec = 5.0 if flip_test else 3.5
    load = 20  # first-call checkpoint load / warm-up
    est = load + secs * per_sec
    return int(max(15, min(cap, est)))


# --------------------------------------------------------------------------------------
# Inference callback (GPU allocated for the duration of this function on ZeroGPU Spaces)
# --------------------------------------------------------------------------------------
@spaces.GPU(duration=_estimate_gpu_duration)
def _recover_gpu(
    video_path: str,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
    hub_repo: str,
    hub_revision: str,
    detector: str,
    pose2d: str,
    backbone: str,
    intrinsics_path: str | None,
    out_dir: str,
    progress: gr.Progress,
):
    """The GPU-bound core (ZeroGPU allocation held ONLY here): load the pipeline + run preproc models +
    the network. Render (moderngl/EGL — not CUDA), npz save, bootstrap and preflight all run OUTSIDE this
    window (in ``run``), so the 60s free-tier budget isn't spent on non-GPU work. Returns the MotionResult
    with ``render=False`` (the pred is torch.saved to ``out_dir`` on the shared FS for ``run`` to render)."""
    from gvhmr.utils.console import progress_hook, progress_phase

    def _hook(frac: float, desc: str) -> None:
        progress(max(0.0, min(1.0, frac)), desc=desc)

    with progress_hook(_hook):
        with progress_phase("Loading model"):
            pipe = _get_pipeline(
                model_repo=hub_repo, revision=hub_revision, detector=detector, pose2d=pose2d, backbone=backbone
            )
        camera = None if static_camera else camera_backend
        # Pass `intrinsics=` ONLY when the user uploaded a sidecar — so a default run never depends on the
        # gvhmr version the Space bootstrapped (the kwarg exists in >=1.1.0; omitting it keeps default runs
        # working if a cold boot lands an older wheel before the release propagates to PyPI).
        extra = {"intrinsics": intrinsics_path} if intrinsics_path else {}
        # GBT: el MotionResult no se puede "picklear" de vuelta desde el worker de ZeroGPU (PicklingError).
        # Se guarda el .npz ACÁ (disco compartido) y se devuelve solo un dict simple.
        res = pipe(
            video_path,
            static_camera=static_camera,
            camera=camera,
            flip_test=flip_test,
            render=False,
            progress=False,
            progress_callback=_hook,
            output_dir=str(out_dir),
            set_overrides=_cpu_preproc_overrides(pose2d, backbone),  # small CPU batch → don't OOM cpu-basic
            **extra,
        )
        # ZeroGPU cannot pickle the rich MotionResult object across its worker boundary.
        # Keep both serializations on the shared filesystem: NPZ for download, full .pt
        # prediction for a faithful post-GPU render of the original 3D mesh.
        npz = res.save_npz(Path(out_dir) / "motion.npz")
        raw_pt = res.save(Path(out_dir) / "hmr4d_prediction.pt")
        return {
            "npz": str(npz),
            "raw_pt": str(raw_pt),
            "staged_video": str(res.video_path),
            "output_dir": str(res.output_dir),
            "num_frames": int(res.num_frames),
            "fps": float(res.fps),
            "camera": str(res.camera),
        }


def _gbt_sumar_articulaciones(npz_path) -> None:
    """GBT: suma al .npz las posiciones 3D de las 22 articulaciones del cuerpo SMPL-X (mundo y cámara) y las
    de reposo (pose T con la forma del cuerpo). Corre en CPU, fuera de la GPU, con el modelo SMPL-X que ya
    bajó el espacio (no hace falta el modelo SMPL aparte)."""
    try:
        import numpy as _gnp
        import smplx
        import torch
        from gvhmr.utils.assets import BODY_MODEL_ROOT

        datos = dict(_gnp.load(npz_path))
        L = int(datos["world_body_pose"].shape[0])
        m = smplx.create(str(BODY_MODEL_ROOT), model_type="smplx", gender="neutral", ext="npz",
                         use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=L)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        with torch.no_grad():
            for marco in ("world", "camera"):
                out = m(global_orient=t(datos[f"{marco}_global_orient"]), body_pose=t(datos[f"{marco}_body_pose"]),
                        betas=t(datos[f"{marco}_betas"]), transl=t(datos[f"{marco}_transl"]))
                datos[f"joints_{marco}"] = out.joints[:, :22].numpy()
            reposo = m(betas=t(datos["world_betas"]))  # pose cero = reposo, misma forma de cuerpo
            datos["joints_rest"] = reposo.joints[0, :22].numpy()
            datos["parents"] = m.parents[:22].numpy()
        _gnp.savez(npz_path, **datos)
        print(f"[gbt] articulaciones agregadas: {L} cuadros")
    except Exception as exc:  # noqa: BLE001 — el .npz de parámetros sigue sirviendo igual
        print(f"[gbt] no se pudieron agregar las articulaciones: {type(exc).__name__}: {exc}")



def _restore_renderable_result(
    video_path: str,
    gpu_result: dict,
    *,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
    detector: str,
    pose2d: str,
    backbone: str,
    intrinsics_path: str | None,
):
    """Rebuild the CPU-side rendering context without recomputing any motion.

    GVHMR MotionResult.render() requires its Hydra config and the staged video,
    alongside the raw prediction. The rich result can't cross a ZeroGPU boundary.
    The same output_dir is reused so build_demo_cfg finds the already-staged 30fps
    input rather than re-running preprocessing.
    """
    import shutil

    import torch
    from gvhmr.cli.demo import build_demo_cfg
    from gvhmr.inference.result import MotionResult

    out_dir = Path(gpu_result["output_dir"])
    overrides = [
        f"output_dir={out_dir}",
        f"detector={detector}",
        f"pose2d={pose2d}",
        f"backbone={backbone}",
        f"camera={camera_backend}",
    ]
    if flip_test:
        overrides.append("flip_test=true")
    cfg = build_demo_cfg(
        Path(video_path), output_root=None, static_cam=static_camera,
        use_dpvo=False, f_mm=None, verbose=False, render_scale=None,
        intrinsics=intrinsics_path, config_overrides=overrides,
    )
    stored_video = Path(gpu_result["staged_video"])
    if not Path(cfg.video_path).is_file() or not stored_video.is_file():
        raise FileNotFoundError(
            f"Staged video missing: config={cfg.video_path}; GPU={stored_video}"
        )
    raw_pt = Path(gpu_result["raw_pt"])
    if not raw_pt.is_file():
        raise FileNotFoundError(f"GPU prediction .pt missing: {raw_pt}")
    expected_pt = Path(cfg.paths.hmr4d_results)
    expected_pt.parent.mkdir(parents=True, exist_ok=True)
    if raw_pt.resolve() != expected_pt.resolve():
        shutil.copyfile(raw_pt, expected_pt)
    # .pt is only read from this very inference request, never a user-supplied file.
    raw = torch.load(raw_pt, map_location="cpu", weights_only=False)
    return MotionResult(
        smpl_params_world=raw["smpl_params_global"],
        smpl_params_camera=raw["smpl_params_incam"],
        intrinsics=raw["K_fullimg"],
        betas_per_frame=raw.get("betas_per_frame"),
        bbx_xys=raw.get("bbx_xys"),
        kp2d=raw.get("kp2d"),
        fps=float(gpu_result["fps"]),
        camera=str(gpu_result["camera"]),
        video_path=stored_video,
        output_dir=out_dir,
        raw=raw,
        _cfg=cfg,
        _device=torch.device("cpu"),
    )


def _tracking_diagnostic(npz_path: str, staged_video: str, output_path: Path) -> Path:
    """OpenCV-only fallback for visualizing 2D detections (NOT a 3D mesh).

    It works even without EGL / moderngl, and makes a useful problem-isolation
    tool: if tracking breaks during an inversion, you'll see it here.
    """
    import cv2
    import numpy as np

    coco_edges = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    ]
    with np.load(npz_path, allow_pickle=False) as data:
        if "kp2d" not in data:
            raise ValueError("NPZ has no 2D keypoints for the diagnostic fallback")
        kp2d = data["kp2d"]
    if kp2d.ndim != 3 or kp2d.shape[2] < 2:
        raise ValueError(f"Unexpected 2D-keypoint shape: {kp2d.shape}")

    cap = cv2.VideoCapture(str(staged_video))
    if not cap.isOpened():
        raise RuntimeError(f"Can't read staged video: {staged_video}")
    writer = None
    scratch = output_path.with_name(output_path.stem + "_mpeg4.mp4")
    frames_written = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Cap the long edge at 960 to keep the diagnostic quick on a CPU Space.
        scale = min(1.0, 960.0 / max(width, height))
        w = max(2, int(width * scale) // 2 * 2)
        h = max(2, int(height * scale) // 2 * 2)
        writer = cv2.VideoWriter(str(scratch), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open a diagnostic video writer")
        for idx in range(len(kp2d)):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            pts = kp2d[idx]
            conf = pts[:, 2] if pts.shape[1] >= 3 else np.ones(len(pts))
            # Points are full-image pixel coordinates in GVHMR.
            def visible(j):
                return j < len(pts) and conf[j] >= 0.25 and np.isfinite(pts[j, :2]).all()
            def xy(j):
                return (int(pts[j, 0] * w / width), int(pts[j, 1] * h / height))
            for a, b in coco_edges:
                if visible(a) and visible(b):
                    cv2.line(frame, xy(a), xy(b), (30, 220, 255), 2, cv2.LINE_AA)
            for j in range(len(pts)):
                if visible(j):
                    cv2.circle(frame, xy(j), 3, (30, 250, 100), -1, cv2.LINE_AA)
            cv2.putText(frame, "2D TRACKING DIAGNOSTIC (not a 3D mesh)",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame {idx + 1}/{len(kp2d)}",
                        (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            frames_written += 1
    finally:
        cap.release()
        if writer:
            writer.release()
    if frames_written == 0:
        raise RuntimeError("No frames written to diagnostic overlay")
    # Browsers prefer H.264/yuv420p to OpenCV's MPEG-4 Part 2 encoding.
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(scratch),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
            check=True, capture_output=True, timeout=300,
        )
        scratch.unlink(missing_ok=True)
    except (OSError, subprocess.SubprocessError):
        scratch.replace(output_path)
    return output_path



def _fetch_smplx_with_diagnostics(progress: gr.Progress, mpi_download, dest: Path) -> None:
    """Official MPI model download with stage logs and actual byte progress.

    No credentials, token values, or credential-bearing URLs are logged.
    """
    import shutil
    import time
    import zipfile
    import requests

    creds = mpi_download.credentials("smplx")
    if not creds:
        raise RuntimeError("SMPL-X credentials missing (SMPLX_USER / SMPLX_PW)")

    spec = mpi_download.ARCHIVES["smplx"]
    print("[gvhmr] SMPL-X: connecting to official MPI download server…", flush=True)
    progress(0.03, desc="SMPL-X: contacting official model server…")
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="gvhmr_smplx_") as td:
        archive = Path(td) / spec["sfile"]
        # 20 s to connect, 45 s between incoming bytes; failures surface instead
        # of indefinitely occupying the Gradio queue.
        with requests.Session() as session:
            with session.post(
                mpi_download.PORTAL,
                params={"domain": spec["domain"], "sfile": spec["sfile"]},
                data={"username": creds[0], "password": creds[1]},
                stream=True,
                allow_redirects=True,
                timeout=(20, 45),
            ) as response:
                status = response.status_code
                content_type = response.headers.get("Content-Type", "").lower()
                total_raw = response.headers.get("Content-Length", "")
                total = int(total_raw) if total_raw.isdecimal() else 0
                print(
                    "[gvhmr] SMPL-X: server responded "
                    f"HTTP {status}, type={content_type or 'unknown'}, "
                    f"content_length={total or 'unknown'}, "
                    f"after={time.monotonic()-t0:.1f}s",
                    flush=True,
                )
                if status in (401, 403):
                    raise PermissionError(
                        f"Official SMPL-X server rejected model access (HTTP {status}). "
                        "Verify email, license acceptance and account credentials."
                    )
                response.raise_for_status()
                if "text/html" in content_type:
                    raise PermissionError(
                        "Official SMPL-X server returned an HTML page, not a model archive. "
                        "The login may be invalid, access may not yet be active, or the portal may have changed."
                    )

                received = 0
                last_logged_mb = -1
                progress(0.05, desc="SMPL-X: receiving model archive…")
                with archive.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        received += len(chunk)
                        whole_mb = received // (10*1024*1024)
                        if whole_mb > last_logged_mb:
                            last_logged_mb = whole_mb
                            print(
                                f"[gvhmr] SMPL-X: received {received/(1024*1024):.1f} MiB"
                                + (f" / {total/(1024*1024):.1f} MiB" if total else ""),
                                flush=True,
                            )
                        if total:
                            frac = min(1.0, received / total)
                            progress(
                                0.05 + 0.09*frac,
                                desc=f"SMPL-X: downloading {frac:.0%} "
                                     f"({received/(1024*1024):.1f} MiB)…",
                            )
                        else:
                            progress(
                                0.08,
                                desc=f"SMPL-X: downloaded {received/(1024*1024):.1f} MiB "
                                     "(size unknown)…",
                            )
                print(
                    f"[gvhmr] SMPL-X: received {received/(1024*1024):.1f} MiB total",
                    flush=True,
                )
        progress(0.14, desc="SMPL-X: validating downloaded archive…")
        if not zipfile.is_zipfile(archive):
            raise RuntimeError(
                "Official download is not a ZIP archive. This commonly indicates "
                "an authentication page, download permission issue, or changed MPI portal response."
            )
        with zipfile.ZipFile(archive) as zf:
            member = next(
                (name for name in zf.namelist()
                 if name.replace("\\", "/").endswith("SMPLX_NEUTRAL.npz")),
                None,
            )
            if member is None:
                raise FileNotFoundError(
                    "SMPLX_NEUTRAL.npz not found in the official model archive. "
                    "Check that the SMPL-X v1.1 models were downloaded."
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(".npz.part")
            try:
                with zf.open(member) as source, part.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024*1024)
                if part.stat().st_size < 100_000:
                    raise RuntimeError("Extracted SMPL-X model is unexpectedly small")
                part.replace(dest)
            finally:
                part.unlink(missing_ok=True)

    print(
        f"[gvhmr] SMPL-X: model ready ({dest.stat().st_size/(1024*1024):.1f} MiB); "
        f"elapsed={time.monotonic()-t0:.1f}s",
        flush=True,
    )
    progress(0.15, desc="SMPL-X body model ready; preparing inference…")


def _prepare_smplx_before_gpu(progress: gr.Progress) -> None:
    """Fetch the registration-gated body model *before* reserving ZeroGPU."""
    from gvhmr.utils import assets, localconfig, mpi_download

    smplx_path = assets.BODY_MODEL_ROOT / "smplx" / "SMPLX_NEUTRAL.npz"
    if smplx_path.is_file():
        print(f"[gvhmr] SMPL-X already present: {smplx_path}", flush=True)
        progress(0.15, desc="SMPL-X body model found; preparing inference…")
        return

    has_user = bool(os.environ.get("SMPLX_USER"))
    has_pw = bool(os.environ.get("SMPLX_PW") or os.environ.get("SMPLX_PASSWORD"))
    mirror = localconfig.body_models_mirror()
    print(
        f"[gvhmr] SMPL-X preflight: username_secret={has_user}, "
        f"password_secret={has_pw}, private_mirror={bool(mirror)}, "
        f"model_present={smplx_path.exists()}",
        flush=True,
    )

    if mirror:
        progress(0.03, desc="SMPL-X: checking private model repository…")
        try:
            assets.ensure_body_models(smpl=False)
        except Exception as exc:
            print(f"[gvhmr] SMPL-X mirror attempt: {type(exc).__name__}: {exc}", flush=True)
        if smplx_path.is_file():
            print("[gvhmr] SMPL-X: obtained from private mirror", flush=True)
            progress(0.15, desc="SMPL-X body model ready; preparing inference…")
            return

    if mpi_download.credentials("smplx") is None:
        raise gr.Error(
            "SMPL-X model missing. Configure a private model mirror "
            "(GVHMR_BODY_MODELS_MIRROR + HF_TOKEN), or add private Space secrets "
            "SMPLX_USER and SMPLX_PW. Never publish credentials or model files."
        )

    try:
        _fetch_smplx_with_diagnostics(progress, mpi_download, smplx_path)
    except Exception as exc:
        print(
            f"[gvhmr] SMPL-X authenticated fetch FAILED: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise gr.Error(
            f"SMPL-X download failed: {type(exc).__name__}: {exc}. "
            "Inspect runtime Logs for HTTP status and bytes received. "
            "Do not share account passwords or tokens."
        ) from exc

    if not smplx_path.is_file():
        raise gr.Error("SMPL-X download completed but model file is still missing.")

def run(
    video_path: str | None,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
    hub_repo: str,
    hub_revision: str,
    detector: str,
    pose2d: str,
    backbone: str,
    native_smplx: bool = True,
    render_overlay: bool = True,
    render_scale: float = 0.25,
    intrinsics_file=None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str, dict, str]:
    """Recover on GPU; restore render context and render on CPU if requested.

    Render errors never discard a successfully computed motion.npz. A 2D
    diagnostic is produced when the 3D mesh renderer fails on headless hardware.
    """
    video_path = _video_path(video_path)
    if not video_path:
        raise gr.Error("Please upload a video first.")
    if video_path.startswith(("http://", "https://")):
        raise gr.Error(
            "The uploaded video could not be read as a local file. Hard-refresh the page, "
            "upload the clip again, and click Recover motion immediately — do not reuse a stale tab."
        )

    def _hook(frac: float, desc: str) -> None:
        progress(max(0.0, min(1.0, frac)), desc=desc)

    try:
        progress(0.0, desc="Preparing environment…")
        _ensure_bootstrapped()
        from gvhmr.utils.console import progress_hook, progress_phase

        progress(0.01, desc="Starting…")
        _prepare_smplx_before_gpu(progress)
        _preflight_stages(
            static_camera=static_camera,
            camera=camera_backend if not static_camera else "simplevo",
            detector=detector,
            pose2d=pose2d,
            backbone=backbone,
        )
        out_dir = Path(tempfile.mkdtemp(prefix="gvhmr_"))
        intrinsics_path = _intrinsics_path(intrinsics_file)
        try:
            result = _recover_gpu(
                video_path, static_camera, camera_backend, flip_test, hub_repo, hub_revision,
                detector, pose2d, backbone, intrinsics_path, str(out_dir), progress,
            )
        except RuntimeError as exc:
            raise gr.Error(f"GVHMR inference failed: {exc}") from exc

        mesh = "smplx" if native_smplx else "smpl"
        npz_path = str(result["npz"])
        overlay_path = None
        overlay_error = None
        fallback_error = None
        overlay_kind = "none"
        with progress_hook(_hook):
            with progress_phase("Saving SMPL params"):
                _gbt_sumar_articulaciones(npz_path)
            if render_overlay:
                with progress_phase("Rendering 3D mesh overlay"):
                    try:
                        restored = _restore_renderable_result(
                            video_path, result,
                            static_camera=static_camera,
                            camera_backend=camera_backend,
                            flip_test=flip_test,
                            detector=detector, pose2d=pose2d, backbone=backbone,
                            intrinsics_path=intrinsics_path,
                        )
                        overlay_path = str(restored.render(
                            out_dir / "overlay_both.mp4", view="both", mesh=mesh,
                            render_scale=float(render_scale),
                        ))
                        overlay_kind = "3D mesh (in-camera | world)"
                        render_status = "ok: reconstructed 3D mesh overlay"
                    except Exception as exc:
                        overlay_error = traceback.format_exc()
                        print("[gvhmr] 3D overlay error:\n" + overlay_error, flush=True)
                        render_status = f"3D renderer failed: {type(exc).__name__}: {exc}"
                        # A diagnostic makes tracking failures inspectable on a CPU-only
                        # Space where an EGL / GL context cannot be created.
                        try:
                            overlay_path = str(_tracking_diagnostic(
                                npz_path, result["staged_video"],
                                out_dir / "tracking_diagnostic.mp4",
                            ))
                            overlay_kind = "2D tracking diagnostic (NOT a 3D mesh)"
                            render_status += "; showing 2D keypoints instead"
                        except Exception as fallback_exc:
                            fallback_error = traceback.format_exc()
                            print("[gvhmr] Diagnostic overlay error:\n" + fallback_error, flush=True)
                            render_status += f"; 2D diagnostic also failed: {fallback_exc}"
            else:
                render_status = "not requested (enable Render overlay video to generate a video)"

        summary = {
            "frames": result["num_frames"],
            "fps": result["fps"],
            "camera": result["camera"],
            "seconds_of_motion": _seconds_of_motion(result["num_frames"], result["fps"]),
            "mesh": "SMPL-X (native)" if native_smplx else "SMPL topology",
            "intrinsics": "custom sidecar" if intrinsics_path else "auto (metadata / FOV heuristic)",
            "overlay": render_status,
            "video_type": overlay_kind,
            "overlay_error": overlay_error.splitlines()[-1] if overlay_error else None,
            "fallback_error": fallback_error.splitlines()[-1] if fallback_error else None,
            "weights_repo": hub_repo.strip(),
            "weights_revision": (hub_revision or "").strip() or "default",
            "detector": detector,
            "pose2d": pose2d,
            "backbone": backbone,
            "render_scale": render_scale,
            "staged_video": result.get("staged_video"),
        }
        debug_path = out_dir / "render_diagnostics.txt"
        debug_path.write_text(
            "GVHMR overlay diagnostic\n"
            f"Video: {video_path}\n"
            f"Output dir: {result.get('output_dir')}\n"
            f"Raw prediction: {result.get('raw_pt')}\n"
            f"Staged video: {result.get('staged_video')}\n"
            f"Summary: {summary!r}\n\n"
            "=== 3D overlay traceback ===\n"
            f"{overlay_error or 'none'}\n\n"
            "=== Diagnostic fallback traceback ===\n"
            f"{fallback_error or 'none'}\n",
            encoding="utf-8",
        )
        progress(1.0, desc="Done")
        return overlay_path, npz_path, summary, str(debug_path)

    except gr.Error:
        raise
    except SystemExit as exc:
        raise gr.Error(
            "GVHMR dependency check failed (see Space Logs for the package list). "
            "On this Space: keep Static camera checked unless you need simplevo, "
            "and use the default vitpose / hmr2 / yolo presets."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — convert everything to a friendly UI error
        if _is_credentials_error(exc):
            raise gr.Error(
                "SMPL/SMPL-X body models are gated and could not be fetched. Configure a private mirror "
                "(GVHMR_BODY_MODELS_MIRROR + HF_TOKEN) or MPI Secrets (SMPLX_USER/SMPLX_PW and, for "
                "rendering, SMPL_USER/SMPL_PW). Details: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise gr.Error(f"Inference failed: {type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def _toggle_camera(static_camera: bool) -> gr.Dropdown:
    """The camera backend is meaningless (and ignored) for a static camera."""
    return gr.Dropdown(interactive=not static_camera)


# Install the runtime deps now, at import — before the platform GPU-schedules the first `run` call.
_startup_bootstrap()


def _probe_gl() -> None:
    """Log whether the moderngl/EGL overlay renderer can init here (diagnostic; the real render is
    best-effort in ``run``). glcontext's standalone EGL needs an *enumerable* device — on a device-less
    cpu-basic box software rendering may be unavailable no matter which Mesa libs are installed, so this
    line tells us (from the Space Logs) which case we're in without running a full job."""
    try:
        import moderngl

        from gvhmr.utils.vis.renderer_gl import _standalone_context

        ctx = _standalone_context(moderngl)
        info = ctx.info
        print(f"[gvhmr] GL overlay context OK — renderer={info.get('GL_RENDERER')!r} ({info.get('GL_VERSION')!r})")
        ctx.release()
    except Exception as exc:  # noqa: BLE001 — purely diagnostic; the app runs (npz-only) either way
        print(f"[gvhmr] GL overlay context UNAVAILABLE ({type(exc).__name__}: {exc}) — overlays fall back to npz-only")


_probe_gl()


def _probe_torch() -> None:
    """Diagnostic: on ZeroGPU the platform preinstalls torch 2.11 and importing pytorch_lightning failed with
    `torch._dynamo ... Config() got an unexpected keyword argument 'deprecated'` (a torch/_config_module vs
    _dynamo skew). Log the exact versions + Config signature + the dynamo/PL import results so we can pin the
    right thing. Runs after the bootstrap so the installed (not platform-preinstall) versions are shown."""
    import importlib
    import inspect

    try:
        import torch

        print(f"[gvhmr] torch {torch.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"[gvhmr] torch import FAIL: {type(exc).__name__}: {exc}")
        return
    for mod in ("torchvision", "pytorch_lightning", "lightning"):
        try:
            print(f"[gvhmr] {mod} {importlib.import_module(mod).__version__} import OK")
        except Exception as exc:  # noqa: BLE001
            print(f"[gvhmr] {mod} import FAIL: {type(exc).__name__}: {exc}")
    try:
        from torch.utils._config_module import Config

        print(f"[gvhmr] _config_module.Config params: {list(inspect.signature(Config).parameters)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[gvhmr] Config introspect FAIL: {type(exc).__name__}: {exc}")
    try:
        importlib.import_module("torch._dynamo")
        print("[gvhmr] torch._dynamo import OK")
    except Exception as exc:  # noqa: BLE001
        print(f"[gvhmr] torch._dynamo import FAIL: {type(exc).__name__}: {exc}")


_probe_torch()


with gr.Blocks(title="GVHMR — Human Motion Recovery") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input video", sources=["upload"], format="mp4")
            static_in = gr.Checkbox(
                value=True,
                label="Static camera",
                info="Recommended on this Space (faster; skips visual odometry). Uncheck only for simplevo.",
            )
            camera_in = gr.Dropdown(
                choices=CAMERA_CHOICES,
                value="simplevo",
                label="Camera backend",
                info="Ignored when 'Static camera' is checked. Scene-aware backends (dpvo/dust3r/vggt) "
                "aren't set up on this Space.",
            )
            flip_in = gr.Checkbox(value=False, label="Flip-test (slower, more accurate)")
            smplx_in = gr.Checkbox(
                value=True,
                label="Native SMPL-X mesh",
                info="Render the full ~10475-vertex SMPL-X surface (the model predicts SMPL-X natively). "
                "Default ON: it needs only the SMPL-X body model. Uncheck for the SMPL-topology mesh — "
                "which additionally requires the gated SMPL body model.",
            )
            render_in = gr.Checkbox(
                value=True,
                label="Render overlay video (slow on CPU)",
                info="ON for debugging: attempt native 3D mesh first, then a clearly labeled "
                "2D keypoint diagnostic if headless OpenGL is unavailable. NPZ is always saved.",
            )

            render_scale_in = gr.Slider(
                minimum=0.25, maximum=1.0, step=0.25, value=0.25,
                label="Overlay render resolution scale (lower = faster)",
            )

            with gr.Accordion("Camera intrinsics (optional)", open=False):
                gr.Markdown(
                    "Upload a **`.json`/`.npz`** sidecar with real intrinsics — `fx`/`fy`/`cx`/`cy` in "
                    "**pixels** (each a scalar or a per-frame list) or a full `K`; optional `width`/`height` "
                    "give the calibration resolution. Fixes world-frame depth/scale + principal point. "
                    "**Leave empty** to auto-detect from video metadata, else fall back to the FOV heuristic. "
                    f"Format: [{REPO_URL}/blob/main/docs/CAMERA_METADATA.md]({REPO_URL}/blob/main/docs/CAMERA_METADATA.md)."
                )
                intrinsics_in = gr.File(
                    label="Intrinsics sidecar (JSON/NPZ) — optional",
                    file_types=[".json", ".npz", ".npy"],
                    file_count="single",
                )

            with gr.Accordion("Model settings", open=False):
                hub_repo_in = gr.Dropdown(
                    choices=_hub_repo_choices(),
                    value=DEFAULT_HUB_REPO,
                    allow_custom_value=True,
                    label="Weights repo (Hub model id)",
                    info="GVHMR checkpoint repo — default is the released weights.",
                )
                hub_revision_in = gr.Textbox(
                    label="Hub revision (optional)",
                    placeholder="main, a tag, or commit hash — blank = default branch",
                )
                detector_in = gr.Dropdown(
                    choices=DETECTOR_CHOICES,
                    value="yolo",
                    allow_custom_value=True,
                    label="Detector",
                    info="All YOLO presets from gvhmr config (`yolo` = yolov8x). Custom weights via name or .pt stem.",
                )
                pose2d_in = gr.Dropdown(
                    choices=POSE2D_CHOICES,
                    value="vitpose",
                    label="2D pose",
                    info="`rtmpose` (RTMPose-m via rtmlib/ONNX) auto-installs on first use; ONNX weights download then too.",
                )
                backbone_in = gr.Dropdown(
                    choices=BACKBONE_CHOICES,
                    value="hmr2",
                    label="Feature backbone",
                    info="`dinov2` only works with a GVHMR checkpoint trained on DINOv2 features.",
                )

            run_btn = gr.Button("Recover motion", variant="primary")

        with gr.Column():
            overlay_out = gr.Video(label="Overlay (in-camera | world)")
            npz_out = gr.File(label="SMPL parameters (.npz)")
            summary_out = gr.JSON(label="Summary")
            debug_out = gr.File(label="Download render diagnostics (.txt)")

    static_in.change(_toggle_camera, inputs=static_in, outputs=camera_in)
    run_btn.click(
        run,
        inputs=[
            video_in,
            static_in,
            camera_in,
            flip_in,
            hub_repo_in,
            hub_revision_in,
            detector_in,
            pose2d_in,
            backbone_in,
            smplx_in,
            render_in,
            render_scale_in,
            intrinsics_in,
        ],
        outputs=[overlay_out, npz_out, summary_out, debug_out],
        show_progress="full",
    )


if __name__ == "__main__":
    # ssr_mode=False is load-bearing on HF Spaces: with SSR on, the browser uploads inputs to the HF Xet
    # CDN and passes a signed URL; gradio then re-fetches it server-side via an *unauthenticated* SSRF-
    # protected GET that 403s (`Failed to download file. Status code: 403`), so uploads never reach the
    # pipeline. Forcing it off here makes uploads go to the Space's own /gradio_api/upload (a local path),
    # deterministically — not reliant on the GRADIO_SSR_MODE env var applying before the container starts.
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        ssr_mode=False,
    )
