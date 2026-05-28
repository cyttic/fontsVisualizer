"""
Glyph refinement pipeline:
  render example → elastic warp → SDXL img2img → trace contours → new font
"""

import os
import array
import threading
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import Glyph as TTGlyph

FONTS_DIR          = os.path.join(os.path.dirname(__file__), "fonts")
MODIFIED_FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts_modified")
MODEL_DIR          = "/mnt/ssd2/cyttic/models/sdxl-refiner-1.0"

PROMPT = (
    "Hebrew handwritten calligraphy letter, angular brush strokes, "
    "white ink on dark navy background, clean sharp edges, high quality"
)
NEG_PROMPT = "blurry, noisy, deformed, extra strokes, low quality, two letters"

_pipe = None
_pipe_lock = threading.Lock()


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_pipe():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                import torch
                from diffusers import StableDiffusionXLImg2ImgPipeline
                print("Loading SDXL refiner…")
                _pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    MODEL_DIR,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
                _pipe.to("cuda")
                _pipe.enable_attention_slicing()
    return _pipe


# ── Render ────────────────────────────────────────────────────────────────────

def render_letter(font_path, letter, size=512):
    img = Image.new("RGB", (size, size), color=(26, 29, 46))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 380)
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), letter, font=font, fill=(226, 232, 240))
    return img


# ── Elastic warp ──────────────────────────────────────────────────────────────

def elastic_warp(img_arr, alpha=80, sigma=12, seed=42):
    rng = np.random.RandomState(seed)
    h, w = img_arr.shape[:2]
    dx = gaussian_filter(rng.randn(h, w) * alpha, sigma)
    dy = gaussian_filter(rng.randn(h, w) * alpha, sigma)
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    cx = np.clip(gx + dx, 0, w - 1)
    cy = np.clip(gy + dy, 0, h - 1)
    out = np.zeros_like(img_arr)
    for c in range(img_arr.shape[2]):
        out[..., c] = map_coordinates(img_arr[..., c], [cy, cx], order=1)
    return out.astype(np.uint8)


# ── SDXL ──────────────────────────────────────────────────────────────────────

def run_sdxl(image, strength=0.40, steps=50, guidance=6.0):
    pipe = get_pipe()
    return pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]


# ── Contour tracing + font insertion ─────────────────────────────────────────

def _signed_area(pts):
    """Signed area in Y-up coordinates. Negative = CW, Positive = CCW."""
    n = len(pts)
    a = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def image_to_font(refined_img, orig_font_path, glyph_name, out_path, fallback_bbox=None):
    """
    Trace contours from refined_img, map to font coordinate space,
    replace glyph in a copy of the font, save to out_path.
    """
    # Threshold
    gray = np.array(refined_img.convert("L"))
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Pixel bbox of the glyph
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        raise ValueError("SDXL output contains no visible glyph")
    px_min, py_min = int(xs.min()), int(ys.min())
    px_max, py_max = int(xs.max()), int(ys.max())

    # Find contours (outer + holes)
    contours_cv, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if not len(contours_cv):
        raise ValueError("No contours found in refined image")

    # Load original font metrics
    font = TTFont(orig_font_path)
    glyf = font["glyf"]
    orig = glyf[glyph_name]
    orig.expand(glyf)   # force parse from raw bytes

    # Fall back to example glyph bbox if the target glyph is empty
    if hasattr(orig, "xMin") and orig.numberOfContours != 0:
        fx_min, fy_min = orig.xMin, orig.yMin
        fx_max, fy_max = orig.xMax, orig.yMax
    elif fallback_bbox:
        fx_min, fy_min, fx_max, fy_max = fallback_bbox
    else:
        head = font["head"]
        fx_min, fy_min = 0, 0
        fx_max, fy_max = head.unitsPerEm, head.unitsPerEm

    def to_font_coords(px, py):
        nx = (px - px_min) / max(px_max - px_min, 1)
        ny = (py - py_min) / max(py_max - py_min, 1)
        return (
            int(round(fx_min + nx * (fx_max - fx_min))),
            int(round(fy_max - ny * (fy_max - fy_min))),  # Y flip
        )

    all_coords = []
    all_flags  = []
    end_pts    = []

    for i, contour in enumerate(contours_cv):
        perim = cv2.arcLength(contour, True)
        eps   = max(1.5, 0.015 * perim)
        approx = cv2.approxPolyDP(contour, eps, True)
        pts_img  = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts_img) < 3:
            continue

        pts_font = [to_font_coords(px, py) for px, py in pts_img]

        # Fix winding: TrueType outer = CW (area < 0), inner = CCW (area > 0)
        is_inner = hierarchy[0][i][3] >= 0
        area = _signed_area(pts_font)
        if not is_inner and area > 0:   # outer but CCW → reverse
            pts_font = pts_font[::-1]
        elif is_inner and area < 0:     # inner but CW → reverse
            pts_font = pts_font[::-1]

        all_coords.extend(pts_font)
        all_flags.extend([1] * len(pts_font))   # all on-curve
        end_pts.append(len(all_coords) - 1)

    if not end_pts:
        raise ValueError("No valid contours after processing")

    # Build new glyph
    from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates

    from fontTools.ttLib.tables.ttProgram import Program

    g = TTGlyph()
    g.numberOfContours    = len(end_pts)
    g.coordinates         = GlyphCoordinates(all_coords)
    g.flags               = array.array("B", all_flags)
    g.endPtsOfContours    = end_pts
    g.xMin = min(x for x, _ in all_coords)
    g.yMin = min(y for _, y in all_coords)
    g.xMax = max(x for x, _ in all_coords)
    g.yMax = max(y for _, y in all_coords)
    g.program = Program()
    g.program.fromAssembly([])   # no hinting instructions

    glyf[glyph_name] = g
    font.save(out_path)
    return out_path


# ── Direct glyph copy (no ML) ────────────────────────────────────────────────

def replace_glyph(target_font, target_cp, example_font, example_cp):
    """Copy glyph from example font directly into target font and save."""
    target_path  = os.path.join(FONTS_DIR, target_font)
    example_path = os.path.join(FONTS_DIR, example_font)

    ef = TTFont(example_path)
    src_name = ef.getBestCmap()[example_cp]
    src_glyph = ef["glyf"][src_name]
    src_glyph.expand(ef["glyf"])

    tf = TTFont(target_path)
    dst_name = tf.getBestCmap()[target_cp]
    tf["glyf"][dst_name] = src_glyph
    # Preserve horizontal metrics from example
    if src_name in ef["hmtx"].metrics and dst_name in tf["hmtx"].metrics:
        tf["hmtx"].metrics[dst_name] = ef["hmtx"].metrics[src_name]
    ef.close()

    out_name = f"{target_font[:-4]}_cp{target_cp:04X}.ttf"
    out_path  = os.path.join(MODIFIED_FONTS_DIR, out_name)
    tf.save(out_path)
    import shutil
    shutil.copy2(out_path, target_path)
    tf.close()
    return out_name


# ── Full job ──────────────────────────────────────────────────────────────────

def run_refine_job(job_id, jobs, target_font, target_cp, example_font, example_cp,
                   alpha=80, sigma=12, seed=42):
    def update(progress, **kw):
        jobs[job_id].update({"progress": progress, **kw})

    try:
        letter = chr(example_cp)

        update(10)
        example_img = render_letter(os.path.join(FONTS_DIR, example_font), letter)

        update(20)
        warped = elastic_warp(np.array(example_img), alpha=alpha, sigma=sigma, seed=seed)

        update(30)
        refined = run_sdxl(Image.fromarray(warped))

        update(80)
        target_path  = os.path.join(FONTS_DIR, target_font)
        example_path = os.path.join(FONTS_DIR, example_font)

        tf = TTFont(target_path)
        glyph_name = tf.getBestCmap()[target_cp]
        tf.close()

        # Get example glyph bbox as fallback for empty target glyphs
        ef = TTFont(example_path)
        eg = ef["glyf"][ef.getBestCmap()[example_cp]]
        eg.expand(ef["glyf"])
        if hasattr(eg, "xMin") and eg.numberOfContours != 0:
            fallback_bbox = (eg.xMin, eg.yMin, eg.xMax, eg.yMax)
        else:
            fallback_bbox = None
        ef.close()

        out_name = f"{target_font[:-4]}_cp{target_cp:04X}.ttf"
        out_path = os.path.join(MODIFIED_FONTS_DIR, out_name)

        image_to_font(refined, target_path, glyph_name, out_path, fallback_bbox=fallback_bbox)

        # Overwrite the original font so the change persists on reload
        import shutil
        shutil.copy2(out_path, target_path)

        update(100, status="done", result_font=out_name, target_font=target_font)

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
