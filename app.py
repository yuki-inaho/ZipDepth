"""
ZipDepth — Gradio demo.

Single-image monocular depth estimation with a MoGe-2 style UI:
    Input image -> colorized depth / A-B compare / assumption-based 3D mesh / downloads.

Run:
    uv run --extra demo python app.py

ZipDepth predicts an *affine-invariant disparity* map (relative depth, scale/shift
undetermined) — not metric depth. The 3D reconstruction tab therefore asks the user
for an assumed FOV and near/far distance and treats them as a choice of affine
transform, not a measurement. See the "3D reconstruction (assumptions)" panel text.
"""

import inspect
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import gradio as gr

from zipdepth.inference.predictor import DepthInference
from zipdepth.utils.colormap import depth_to_colormap

# --------------------------------------------------------------------------------------
# Compare tab: gr.ImageSlider only exists on Gradio 5+ (this app targets Gradio 6, which
# has it). We keep a plain two-image Row as a fallback in case an older Gradio without
# it is ever installed. The branch picked here changes how many outputs the submit
# handler must return, so HAS_IMAGESLIDER is read once at import time and reused
# everywhere `.click(outputs=...)` is wired up.
# --------------------------------------------------------------------------------------
HAS_IMAGESLIDER = hasattr(gr, "ImageSlider")


REPO_ROOT = Path(__file__).resolve().parent

CHECKPOINTS = {
    "zipdepth_base (GPU)": {
        "path": str(REPO_ROOT / "checkpoints" / "zipdepth_base.pth"),
        "upsample_unfold": True,
    },
    "zipdepth_base_npu (NPU/mobile)": {
        "path": str(REPO_ROOT / "checkpoints" / "zipdepth_base_npu.pth"),
        "upsample_unfold": False,
    },
}

_EXAMPLE_CANDIDATES = [
    REPO_ROOT / "assets" / "examples" / "im0.jpg",
    REPO_ROOT / "assets" / "qualitative" / "driving_night_rgb.jpg",
    REPO_ROOT / "assets" / "qualitative" / "car_show_rgb.jpg",
    REPO_ROOT / "assets" / "qualitative" / "close_up_rgb.jpg",
    REPO_ROOT / "assets" / "qualitative" / "synthetic_indoor_rgb.jpg",
]
EXAMPLE_PATHS = [str(p) for p in _EXAMPLE_CANDIDATES if p.exists()]


# --------------------------------------------------------------------------------------
# Device probe. torch.cuda.is_available() alone is not enough on this machine
# (GTX 1070 / sm_61): recent PyPI CUDA wheels drop Pascal kernels, so a conv can still
# raise "no kernel image is available for execution on the device" at runtime.
# Actually run a tiny conv to find out, and fall back to CPU if it fails.
# --------------------------------------------------------------------------------------
def _probe_device() -> tuple:
    """Returns (device, human_readable_note)."""
    if not torch.cuda.is_available():
        return "cpu", "CUDA が利用できないため CPU で実行します"
    try:
        x = torch.randn(1, 3, 32, 32, device="cuda")
        c = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
        _ = c(x)
        torch.cuda.synchronize()
        return "cuda", f"GPU: {torch.cuda.get_device_name(0)}"
    except Exception as e:
        return "cpu", f"GPU が使用できないため CPU にフォールバック: {e}"


DEVICE, DEVICE_NOTE = _probe_device()

# Pascal (compute capability 6.x — GTX 10xx / GP10x) has crippled FP16 throughput
# (~1/64 of FP32, since it lacks real FP16 ALUs), so the FP16 checkbox would make
# inference *slower*, not faster, on this class of GPU. Warn instead of silently
# disabling it — the box is still usable, just not a speed win here.
_PASCAL_FP16_WARNING = None
if DEVICE == "cuda":
    try:
        major, _minor = torch.cuda.get_device_capability(0)
        if major == 6:
            _PASCAL_FP16_WARNING = (
                "この GPU は Pascal 世代 (compute capability 6.x) のため FP16 は "
                "FP32 よりも遅くなります（FP16 スループットが FP32 の約1/64）。"
                "ここでは速度目的では ON にしないでください。"
            )
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Model cache. DepthInference reuses shared mutable buffers (_resize_buf, _gpu_buf_*)
# without any internal locking, so concurrent requests against the *same* instance can
# corrupt each other's buffers or crash on a shape mismatch. We therefore:
#   - cache one DepthInference per (checkpoint_path, upsample_unfold, use_half),
#   - hand out one threading.Lock per cached instance, and
#   - require callers to hold that lock from the `input_size` assignment through the
#     end of `infer_image()`.
# `input_size` is deliberately NOT part of the cache key: `_ensure_buffers` reallocates
# buffers on demand when resolution changes, so rebuilding the whole model per slider
# move would just be wasteful.
# --------------------------------------------------------------------------------------
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_predictor(checkpoint_path: str, upsample_unfold: bool, use_half: bool):
    key = (checkpoint_path, upsample_unfold, use_half)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            predictor = DepthInference(
                checkpoint_path=checkpoint_path,
                variant="base",
                device=DEVICE,
                use_half=use_half,
                use_compile=False,
                input_size=384,
                warmup_iters=3,
                upsample_unfold=upsample_unfold,
            )
            _MODEL_CACHE[key] = (predictor, threading.Lock())
        return _MODEL_CACHE[key]


def _round_to_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


# --------------------------------------------------------------------------------------
# 3D reconstruction. ZipDepth only gives affine-invariant disparity (no absolute scale).
# We ask the user for an assumed FOV + near/far distance and treat that as *choosing*
# the otherwise-undetermined affine transform, then unproject with a pinhole model.
# --------------------------------------------------------------------------------------
def build_mesh_glb(
    rgb: np.ndarray,
    depth: np.ndarray,
    fov_deg: float,
    near_m: float,
    far_m: float,
    mesh_res: int,
    edge_thresh: float,
    bg_cutoff: float,
    out_path: Path,
) -> str:
    h, w = depth.shape
    scale = mesh_res / max(h, w)
    if scale < 1.0:
        new_w = max(2, int(round(w * scale)))
        new_h = max(2, int(round(h * scale)))
        depth_r = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb_r = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_r, rgb_r = depth, rgb
    h, w = depth_r.shape

    # disparity -> [0, 1], 1 = nearest
    d_min, d_max = float(depth_r.min()), float(depth_r.max())
    d_norm = (depth_r - d_min) / (d_max - d_min + 1e-8)

    # Treat near/far as the endpoints of the affine map in *disparity* (1/depth) space.
    inv_near = 1.0 / max(float(near_m), 1e-6)
    inv_far = 1.0 / max(float(far_m), 1e-6)
    inv_depth = d_norm * (inv_near - inv_far) + inv_far
    inv_depth = np.clip(inv_depth, 1e-6, None)
    depth_m = 1.0 / inv_depth

    fx = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    fy = fx
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    X = (us - w / 2.0) / fx * depth_m
    Y = (vs - h / 2.0) / fy * depth_m
    Z = depth_m

    # glTF/.glb uses an OpenGL-style right-handed coordinate system (Y up, camera
    # looking down -Z); our image-space unprojection has Y down and Z forward, so both
    # need flipping or the mesh comes out upside-down and back-to-front.
    verts = np.stack([X, -Y, -Z], axis=-1).reshape(-1, 3).astype(np.float32)
    colors = np.ascontiguousarray(rgb_r).reshape(-1, 3).astype(np.uint8)

    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    v00 = idx[:-1, :-1]
    v01 = idx[:-1, 1:]
    v10 = idx[1:, :-1]
    v11 = idx[1:, 1:]

    # Drop triangles that straddle a depth discontinuity ("remove edges" à la MoGe):
    # a cell is kept only if its 4 corner disparities are all within edge_thresh of
    # each other. Also drop cells that are pure far background: sky/distant scenery
    # sits at disparity ≈ 0, and left unfiltered it becomes a huge flat backdrop that
    # dwarfs the actual subject once near/far are applied. bg_cutoff=0 disables this
    # (keeps everything, same as before).
    corners = np.stack([d_norm[:-1, :-1], d_norm[:-1, 1:], d_norm[1:, :-1], d_norm[1:, 1:]], axis=0)
    corner_max = corners.max(axis=0)
    cell_spread = corner_max - corners.min(axis=0)
    valid = ((cell_spread <= edge_thresh) & (corner_max > bg_cutoff)).reshape(-1)

    tri1 = np.stack([v00, v10, v01], axis=-1).reshape(-1, 3)
    tri2 = np.stack([v10, v11, v01], axis=-1).reshape(-1, 3)
    faces = np.concatenate([tri1[valid], tri2[valid]], axis=0)

    if faces.shape[0] == 0:
        raise ValueError(
            "edge_thresh が小さすぎるか bg_cutoff が大きすぎて、全ての三角形が除去されました。"
            "値を調整してください"
        )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)

    # Winding is correct by construction — verified analytically, not by fix_normals().
    # Locally approximating depth as constant, with u increasing rightward and v
    # increasing downward, a grid step k>0 gives dX/du=+k, dY_flipped/dv=-k, Z=-depth.
    # For tri1=[v00,v10,v01]=[(u,v),(u,v+1),(u+1,v)]: e1=(0,-k,0), e2=(k,0,0),
    # e1×e2=(0,0,+k^2). Same for tri2=[v10,v11,v01]. Camera sits at the origin looking
    # down -Z and the geometry is at Z<0, so a +Z normal faces the camera: front-facing.
    # Do NOT call mesh.fix_normals() here — for an open (non-watertight) surface like
    # this height-field it re-orients faces with a volume-based heuristic that can flip
    # this already-correct winding, and it's a slow O(faces) connected-component pass
    # at mesh_res up to 640 (~4×10^5 faces).
    mesh.remove_unreferenced_vertices()  # edge_thresh drops faces but not their verts

    mesh.export(str(out_path))
    return str(out_path)


_TEMP_DIR_PREFIX = "zipdepth_"
_MAX_TEMP_DIRS = 20


def _prune_old_temp_dirs(keep: int = _MAX_TEMP_DIRS) -> None:
    """Each request makes its own tempfile.mkdtemp() output dir that is never cleaned
    up, so a long-running server slowly fills /tmp. Keep only the `keep` most recently
    modified zipdepth_* dirs and delete the rest. Best-effort — a failed rmtree (e.g.
    a file another process still has open) is silently ignored."""
    base = Path(tempfile.gettempdir())
    dirs = sorted(
        (p for p in base.glob(f"{_TEMP_DIR_PREFIX}*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_dir in dirs[keep:]:
        shutil.rmtree(old_dir, ignore_errors=True)


def _make_model3d(**kwargs) -> gr.Model3D:
    """gr.Model3D's kwargs (display_mode, clear_color, ...) vary across Gradio
    versions; keep only the ones the installed version actually accepts."""
    valid_params = inspect.signature(gr.Model3D.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return gr.Model3D(**filtered)


# --------------------------------------------------------------------------------------
# Main callback
# --------------------------------------------------------------------------------------
def on_submit(
    input_image,
    checkpoint_choice,
    input_size,
    colormap_name,
    invert_cmap,
    use_fp16,
    fov_deg,
    near_m,
    far_m,
    mesh_res,
    edge_thresh,
    bg_cutoff,
):
    if input_image is None:
        raise gr.Error("画像をアップロードしてください")

    rgb = np.asarray(input_image)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    # Gradio gives RGB (numpy), DepthInference.infer_image expects BGR.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])

    cfg = CHECKPOINTS[checkpoint_choice]
    use_half = bool(use_fp16) and DEVICE == "cuda"
    predictor, lock = get_predictor(cfg["path"], cfg["upsample_unfold"], use_half)

    size = _round_to_multiple(float(input_size), 32)
    h0, w0 = bgr.shape[:2]

    t0 = time.time()
    with lock:
        predictor.input_size = size
        model_h, model_w = predictor._compute_target_size(h0, w0)
        depth = predictor.infer_image(bgr)
    elapsed_ms = (time.time() - t0) * 1000.0

    # depth_to_colormap returns BGR (cv2 convention); Gradio wants RGB.
    depth_bgr = depth_to_colormap(depth, cmap=colormap_name, invert=bool(invert_cmap))
    depth_rgb = np.ascontiguousarray(depth_bgr[:, :, ::-1])

    h, w = depth.shape
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
    stats_md = (
        f"**Resolution**: {w}×{h} (original) → model input {model_w}×{model_h} "
        f"(requested shorter side {size}px)\n\n"
        f"**Inference time**: {elapsed_ms:.1f} ms ({fps:.1f} FPS, single-shot)\n\n"
        f"**Value range** (disparity — larger = closer): [{depth.min():.4f}, {depth.max():.4f}]\n\n"
        f"**Device**: {DEVICE_NOTE}"
    )

    out_dir = Path(tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX))
    stem = "zipdepth_output"

    color_path = out_dir / f"{stem}_depth_color.png"
    cv2.imwrite(str(color_path), depth_bgr)

    # These two are disparity (larger = closer), not depth — name them accordingly so
    # downstream consumers don't mistake them for metric depth.
    d_min, d_max = float(depth.min()), float(depth.max())
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    disparity_16bit = (depth_norm * 65535.0).astype(np.uint16)
    gray_path = out_dir / f"{stem}_disparity_16bit.png"
    cv2.imwrite(str(gray_path), disparity_16bit)

    raw_path = out_dir / f"{stem}_disparity_raw.npy"
    np.save(raw_path, depth)

    download_files = [str(color_path), str(gray_path), str(raw_path)]

    model3d_path = None
    try:
        mesh_path = out_dir / f"{stem}_mesh.glb"
        model3d_path = build_mesh_glb(
            rgb, depth,
            fov_deg=float(fov_deg), near_m=float(near_m), far_m=float(far_m),
            mesh_res=int(mesh_res), edge_thresh=float(edge_thresh),
            bg_cutoff=float(bg_cutoff),
            out_path=mesh_path,
        )
        download_files.append(model3d_path)
        note_md = (
            "ZipDepth outputs only relative (affine-invariant) disparity — there is no "
            "absolute scale. The mesh above is built by treating the assumed FOV / "
            "near / far values as a **choice** of the otherwise-undetermined affine "
            "transform, not a measurement. Change them in the Settings panel and "
            "re-submit to reshape the point cloud."
        )
    except Exception as e:
        note_md = f"3D メッシュの生成に失敗しました: {e}"

    _prune_old_temp_dirs()

    if HAS_IMAGESLIDER:
        return depth_rgb, stats_md, (rgb, depth_rgb), model3d_path, note_md, download_files
    return depth_rgb, stats_md, rgb, depth_rgb, model3d_path, note_md, download_files


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
HEADER_MD = """
<div align="center">
  <h1>⚡ ZipDepth — Lightweight Zero-Shot Monocular Depth</h1>
  <p>
    <a href="https://arxiv.org/abs/2607.08771" style="text-decoration:none"><img alt="Paper" style="display:inline-block;vertical-align:middle" src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
    <a href="https://zipdepth.github.io/" style="text-decoration:none"><img alt="Project Page" style="display:inline-block;vertical-align:middle" src="https://img.shields.io/badge/Project-Page-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white"></a>
    <a href="https://github.com/fabiotosi92/ZipDepth" style="text-decoration:none"><img alt="GitHub" style="display:inline-block;vertical-align:middle" src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white"></a>
  </p>
</div>

ZipDepth predicts **relative (affine-invariant) disparity**, not metric depth — the 3D
tab asks you to supply an assumed FOV / near / far distance rather than claiming to
recover a real-world scale.
"""

with gr.Blocks(title="ZipDepth") as demo:
    gr.Markdown(HEADER_MD)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", image_mode="RGB", label="Input Image")

            with gr.Accordion("Settings", open=False):
                checkpoint_choice = gr.Dropdown(
                    choices=list(CHECKPOINTS.keys()),
                    value=next(iter(CHECKPOINTS)),
                    label="Checkpoint",
                )
                input_size = gr.Slider(
                    256, 1024, value=384, step=32,
                    label="Inference resolution (shorter side)",
                    info="Rounded to a multiple of 32. Larger = sharper but slower.",
                )
                colormap_name = gr.Dropdown(
                    ["Spectral", "turbo", "viridis", "magma", "inferno", "gray"],
                    value="Spectral",
                    label="Colormap",
                )
                invert_cmap = gr.Checkbox(value=True, label="Invert colormap (near = red)")
                use_fp16 = gr.Checkbox(
                    value=False, label="FP16 (CUDA only)",
                    interactive=(DEVICE == "cuda"),
                    info=_PASCAL_FP16_WARNING,
                )
                gr.Markdown(f"🖥️ {DEVICE_NOTE}")

            with gr.Accordion("3D reconstruction (assumptions)", open=False):
                gr.Markdown(
                    "ZipDepth only outputs relative disparity. The values below are "
                    "**assumptions you provide**, not measurements — they choose the "
                    "affine transform (scale + shift) the model cannot determine on "
                    "its own."
                )
                fov_deg = gr.Slider(20, 120, value=55, step=1, label="Assumed horizontal FOV (deg)")
                near_m = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Assumed near distance (m)")
                far_m = gr.Slider(2.0, 100.0, value=5.0, step=0.5, label="Assumed far distance (m)")
                mesh_res = gr.Slider(128, 640, value=384, step=32, label="3D mesh resolution (longer side)")
                edge_thresh = gr.Slider(
                    0.005, 0.2, value=0.04, step=0.005,
                    label="Edge removal threshold",
                    info="Triangles whose corners span more than this normalized disparity gap are dropped.",
                )
                bg_cutoff = gr.Slider(
                    0.0, 0.3, value=0.02, step=0.01,
                    label="Drop far background below this disparity",
                    info="Sky and distant background sit at disparity ≈ 0 and become a huge "
                         "backdrop. 0 keeps everything.",
                )

            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Depth"):
                    depth_image = gr.Image(type="numpy", label="Colorized Depth", format="png")
                    stats_md = gr.Markdown()

                with gr.Tab("Compare"):
                    if HAS_IMAGESLIDER:
                        compare_slider = gr.ImageSlider(label="Input vs Depth", type="numpy")
                        compare_outputs = [compare_slider]
                    else:
                        with gr.Row():
                            compare_img1 = gr.Image(type="numpy", label="Input", interactive=False)
                            compare_img2 = gr.Image(type="numpy", label="Depth", interactive=False)
                        compare_outputs = [compare_img1, compare_img2]

                with gr.Tab("3D View"):
                    model3d = _make_model3d(
                        display_mode="solid",
                        clear_color=[1, 1, 1, 1],
                        height="60vh",
                        label="Point cloud mesh (relative, assumption-based)",
                        # Note: gr.Model3D's camera_position had no measurable effect on the
                        # initial framing here (verified by before/after screenshots on
                        # Gradio 6.24), so it is deliberately not set rather than left in as
                        # a no-op that looks like it does something.
                    )
                    note_md = gr.Markdown()

                with gr.Tab("Download"):
                    files = gr.File(
                        type="filepath",
                        label="Output Files — disparity outputs are normalized disparity (larger = closer), not metric depth",
                        file_count="multiple",
                    )

    if EXAMPLE_PATHS:
        gr.Examples(examples=[[p] for p in EXAMPLE_PATHS], inputs=[input_image])

    submit_btn.click(
        fn=on_submit,
        inputs=[
            input_image, checkpoint_choice, input_size, colormap_name, invert_cmap, use_fp16,
            fov_deg, near_m, far_m, mesh_res, edge_thresh, bg_cutoff,
        ],
        outputs=[depth_image, stats_md] + compare_outputs + [model3d, note_md, files],
    )


if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", "7860"))
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=port, share=False, theme=gr.themes.Soft()
    )
