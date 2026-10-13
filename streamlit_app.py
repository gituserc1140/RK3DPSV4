import os
import tempfile
import traceback
import io
import math
import base64
import html
import json
import re
import uuid

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import trimesh
from dotenv import load_dotenv
from PIL import Image, ImageDraw


load_dotenv()

DEFAULT_SOCIAL_CAPTION = "Fresh 3D concept created in this app."
DEFAULT_SOCIAL_TAGS = "3d, 3dmodel, digitalart, tiktok"
DEFAULT_SOCIAL_CTA = "What should I create next?"
SOCIAL_POST_FORMAT_INTROS = {
    "Showcase": "✨ New 3D concept reveal",
    "Behind the scenes": "🛠️ Behind the build",
    "Shop promo": "🛍️ New product-style mockup",
}


def get_api_key(key_name):
    """Get API key from environment or Streamlit secrets."""
    # Re-read local .env so rotated keys are picked up without a full app restart.
    load_dotenv(override=True)
    value = os.environ.get(key_name)
    if value:
        return value

    try:
        return st.secrets[key_name]
    except Exception:
        return None


_COHERE_SESSION_KEYS = ["cohere_api_key_prompt", "cohere_api_key_chat", "cohere_api_key_blog"]


def get_cohere_api_key():
    """Get the session-provided or configured Cohere API key."""
    for session_key_name in _COHERE_SESSION_KEYS:
        session_key = st.session_state.get(session_key_name, "").strip()
        if session_key:
            return session_key

    candidate_names = [
        "COHERE_API_KEY",
        "COHERE_KEY",
        "RKStudioCohere",
        "RKSTUDIO_COHERE_API_KEY",
    ]
    for name in candidate_names:
        value = get_api_key(name)
        if value:
            return value
    return None


def render_cohere_api_key_input(session_key_name):
    """Render the Cohere API key field within the current tab."""
    st.text_input(
        "Cohere API key",
        type="password",
        key=session_key_name,
        help="Used only for this browser session and not saved to this app's configuration.",
    )


def get_cohere_response(api_key, user_input, history, model_name, temperature, max_tokens):
    """Generate a Cohere response, supporting both V1 and V2 SDK clients."""
    import cohere

    # Try V1 first.
    client = cohere.Client(api_key)
    v1_history = [
        {
            "role": "USER" if msg["role"] == "user" else "CHATBOT",
            "message": msg["content"],
        }
        for msg in history
    ]
    try:
        response = client.chat(
            message=user_input,
            chat_history=v1_history,
            model=model_name,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        text = getattr(response, "text", "")
        if text:
            return text.strip()
    except Exception:
        pass

    # Fallback to V2.
    client_v2 = cohere.ClientV2(api_key)
    messages = [
        {
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"],
        }
        for msg in history
    ]
    messages.append({"role": "user", "content": user_input})
    response_v2 = client_v2.chat(
        model=model_name,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )

    if hasattr(response_v2, "message") and getattr(response_v2.message, "content", None):
        parts = []
        for block in response_v2.message.content:
            if getattr(block, "type", "") == "text":
                parts.append(getattr(block, "text", ""))
        text = "\n".join([part for part in parts if part]).strip()
        if text:
            return text

    raise ValueError("No response text returned by Cohere")


def _project_point_iso(point):
    """Project a 3D point into a 2D isometric-like view."""
    x, y, z = point
    cos_30 = math.sqrt(3) / 2
    sin_30 = 0.5
    return ((x - y) * cos_30, (x + y) * sin_30 - z)


def build_static_snapshot(glb_path, note=""):
    """Render a static PNG snapshot from mesh vertices when triangulation fails."""
    width, height = 960, 640
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)

    loaded = trimesh.load(glb_path, force="scene")
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]

    points = []
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        vertices = getattr(mesh, "vertices", None)
        if vertices is None or len(vertices) == 0:
            continue

        sample_step = max(1, len(vertices) // 2500)
        for point in vertices[::sample_step]:
            points.append(point)

    if not points:
        draw.rectangle((28, 28, width - 28, height - 28), outline="#94a3b8", width=2)
        draw.text((44, 44), "Static snapshot unavailable", fill="#0f172a")
        if note:
            draw.text((44, 74), f"Reason: {note[:120]}", fill="#334155")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    projected = [_project_point_iso(point) for point in points]
    xs = [pt[0] for pt in projected]
    ys = [pt[1] for pt in projected]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(max_x - min_x, 1e-9)
    dy = max(max_y - min_y, 1e-9)

    margin = 56
    sx = (width - (2 * margin)) / dx
    sy = (height - (2 * margin)) / dy
    scale = min(sx, sy)

    draw.rectangle((24, 24, width - 24, height - 24), outline="#94a3b8", width=2)
    for x, y in projected:
        px = margin + (x - min_x) * scale
        py = margin + (y - min_y) * scale
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill="#0f172a")

    draw.text((36, 34), "Static snapshot fallback", fill="#0f172a")
    if note:
        draw.text((36, 58), f"Reason: {note[:120]}", fill="#334155")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_placeholder_snapshot(note=""):
    """Return a simple placeholder PNG when model snapshot generation fails."""
    width, height = 960, 640
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, width - 24, height - 24), outline="#94a3b8", width=2)
    draw.text((40, 44), "Model image preview unavailable", fill="#0f172a")
    if note:
        draw.text((40, 74), f"Reason: {note[:120]}", fill="#334155")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def update_snapshot_preview(glb_path):
    """Store a PNG snapshot preview in session for reliable display."""
    try:
        st.session_state["glb_preview_png"] = build_static_snapshot(glb_path)
        st.session_state["glb_preview_error"] = ""
    except Exception as exc:
        st.session_state["glb_preview_png"] = build_placeholder_snapshot(str(exc))
        st.session_state["glb_preview_error"] = str(exc)


def show_preview_mode_badge(mode):
    """Display a compact badge indicating the active preview mode."""
    palette = {
        "Interactive": ("#166534", "#dcfce7", "#bbf7d0"),
        "Physical": ("#991b1b", "#fee2e2", "#fecaca"),
        "Static": ("#7c2d12", "#ffedd5", "#fed7aa"),
    }
    text_color, bg_color, border_color = palette.get(mode, ("#0f172a", "#e2e8f0", "#cbd5e1"))
    st.markdown(
        (
            f"<div style='display:inline-block;padding:0.15rem 0.55rem;border-radius:999px;"
            f"font-size:0.78rem;font-weight:600;color:{text_color};background:{bg_color};"
            f"border:1px solid {border_color};margin-bottom:0.35rem;'>"
            f"Preview: {mode}</div>"
        ),
        unsafe_allow_html=True,
    )


def get_default_viewer_settings():
    """Return default interactive viewer settings."""
    return {
        "max_faces": 60000,
        "normalize": False,
        "wireframe": False,
        "wireframe_limit": 12000,
        "ambient": 0.44,
        "diffuse": 0.48,
        "specular": 0.04,
        "roughness": 0.84,
        "flatshading": False,
        "bg_color": "#ffffff",
        "camera_preset": "Isometric",
        "projection": "perspective",
        "show_axes": False,
    }


def _camera_eye_from_preset(name):
    """Map a friendly camera preset name to a Plotly camera eye vector."""
    presets = {
        "Isometric": {"x": 1.65, "y": 1.65, "z": 1.0},
        "Front": {"x": 0.0, "y": 2.5, "z": 0.2},
        "Top": {"x": 0.01, "y": 0.01, "z": 2.8},
        "Left": {"x": -2.5, "y": 0.1, "z": 0.2},
    }
    return presets.get(name, presets["Isometric"])


def _downsample_faces(faces, max_faces):
    """Reduce face count deterministically for smoother browser rendering."""
    if len(faces) <= max_faces:
        return faces
    step = max(1, int(math.ceil(len(faces) / max_faces)))
    sampled = faces[::step]
    if len(sampled) > max_faces:
        sampled = sampled[:max_faces]
    return sampled


def _normalize_vertices(vertices):
    """Center and scale vertices to keep camera framing consistent across models."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float((maxs - mins).max())
    if extent <= 1e-9:
        extent = 1.0
    return (vertices - center) / extent


def _to_plotly_rgba_colors(colors):
    """Convert trimesh color arrays into Plotly-friendly rgba strings."""
    if colors is None:
        return None

    array = np.asarray(colors)
    if array.ndim != 2 or array.shape[1] < 3 or len(array) == 0:
        return None

    rgb = array[:, :3].astype(float)
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0:
        rgb = np.clip(rgb * 255.0, 0, 255)

    if array.shape[1] >= 4:
        alpha = array[:, 3].astype(float)
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(alpha)) <= 1.0:
            alpha = np.clip(alpha * 255.0, 0, 255)
    else:
        alpha = np.full((len(array),), 255.0)

    return [
        f"rgba({int(r)},{int(g)},{int(b)},{float(a) / 255.0:.3f})"
        for (r, g, b), a in zip(rgb, alpha)
    ]


def _build_wireframe_trace(vertices, faces, edge_limit):
    """Build an optional wireframe overlay trace with bounded edge count."""
    edges = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (int(u), int(v)) if u <= v else (int(v), int(u))
            if edge not in edges:
                edges.add(edge)
            if len(edges) >= edge_limit:
                break
        if len(edges) >= edge_limit:
            break

    if not edges:
        return None

    xs = []
    ys = []
    zs = []
    for u, v in edges:
        xs.extend([vertices[u, 0], vertices[v, 0], None])
        ys.extend([vertices[u, 1], vertices[v, 1], None])
        zs.extend([vertices[u, 2], vertices[v, 2], None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "#111827", "width": 2},
        hoverinfo="skip",
        showlegend=False,
        name="wireframe",
    )


def build_plotly_figure(glb_path, settings=None):
    """Build a Plotly mesh figure from a GLB file."""
    settings = {**get_default_viewer_settings(), **(settings or {})}
    loaded = trimesh.load(glb_path, force="scene")
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]

    figure = go.Figure()
    has_geometry = False
    failed_meshes = 0

    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces is None or len(mesh.faces) == 0:
            continue

        try:
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces, dtype=int)

            if vertices.size == 0 or faces.size == 0:
                failed_meshes += 1
                continue

            faces = _downsample_faces(faces, int(settings["max_faces"]))

            used = np.unique(faces.reshape(-1))
            compact_vertices = vertices[used]
            remap = {int(old): idx for idx, old in enumerate(used)}
            compact_faces = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces], dtype=int)

            if settings.get("normalize", True):
                compact_vertices = _normalize_vertices(compact_vertices)
        except Exception:
            failed_meshes += 1
            continue

        i = compact_faces[:, 0]
        j = compact_faces[:, 1]
        k = compact_faces[:, 2]
        color = "#b8b8b8"
        vertex_colors = None
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None) if visual is not None else None
        raw_vertex_colors = getattr(visual, "vertex_colors", None) if visual is not None else None
        if raw_vertex_colors is not None:
            try:
                vertex_array = np.asarray(raw_vertex_colors)
                if vertex_array.ndim == 2 and len(vertex_array) == len(vertices):
                    vertex_colors = _to_plotly_rgba_colors(vertex_array[used])
            except Exception:
                vertex_colors = None
        base_color = getattr(material, "baseColorFactor", None)
        if base_color and len(base_color) >= 3:
            rgb = tuple(int(channel) for channel in base_color[:3])
            color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        mesh_trace_kwargs = {
            "x": compact_vertices[:, 0],
            "y": compact_vertices[:, 1],
            "z": compact_vertices[:, 2],
            "i": i,
            "j": j,
            "k": k,
            "flatshading": bool(settings["flatshading"]),
            "lighting": {
                "ambient": float(settings["ambient"]),
                "diffuse": float(settings["diffuse"]),
                "specular": float(settings["specular"]),
                "roughness": float(settings["roughness"]),
            },
            "lightposition": {"x": 80, "y": 80, "z": 130},
            "hoverinfo": "skip",
            "name": os.path.basename(glb_path),
            "showscale": False,
        }
        if vertex_colors:
            mesh_trace_kwargs["vertexcolor"] = vertex_colors
        else:
            mesh_trace_kwargs["color"] = color

        figure.add_trace(go.Mesh3d(**mesh_trace_kwargs))

        if settings.get("wireframe"):
            wire = _build_wireframe_trace(
                compact_vertices,
                compact_faces,
                edge_limit=int(settings["wireframe_limit"]),
            )
            if wire is not None:
                figure.add_trace(wire)

        has_geometry = True

    if not has_geometry:
        if failed_meshes:
            raise ValueError("GLB mesh could not be triangulated cleanly")
        raise ValueError("No mesh geometry found in GLB file")

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=settings["bg_color"],
        scene={
            "bgcolor": settings["bg_color"],
            "aspectmode": "data",
            "xaxis": {"visible": bool(settings["show_axes"]), "showgrid": False, "zeroline": False},
            "yaxis": {"visible": bool(settings["show_axes"]), "showgrid": False, "zeroline": False},
            "zaxis": {"visible": bool(settings["show_axes"]), "showgrid": False, "zeroline": False},
            "camera": {
                "eye": _camera_eye_from_preset(settings["camera_preset"]),
                "projection": {"type": settings["projection"]},
            },
        },
        showlegend=False,
    )
    return figure


def display_3d_model(glb_path, chart_key=None):
    """Display a 3D GLB model using Plotly as a browser-safe fallback."""
    st.caption(f"Viewing: {os.path.basename(glb_path)}")

    viewer_key = chart_key or "viewer"

    show_preview_mode_badge("Interactive")

    defaults = get_default_viewer_settings()
    settings = {
        "max_faces": st.session_state.get(f"{viewer_key}_max_faces", defaults["max_faces"]),
        "normalize": st.session_state.get(f"{viewer_key}_normalize", defaults["normalize"]),
        "wireframe": st.session_state.get(f"{viewer_key}_wireframe", defaults["wireframe"]),
        "wireframe_limit": st.session_state.get(f"{viewer_key}_wireframe_limit", defaults["wireframe_limit"]),
        "ambient": st.session_state.get(f"{viewer_key}_ambient", defaults["ambient"]),
        "diffuse": st.session_state.get(f"{viewer_key}_diffuse", defaults["diffuse"]),
        "specular": st.session_state.get(f"{viewer_key}_specular", defaults["specular"]),
        "roughness": st.session_state.get(f"{viewer_key}_roughness", defaults["roughness"]),
        "flatshading": st.session_state.get(f"{viewer_key}_flatshading", defaults["flatshading"]),
        "bg_color": st.session_state.get(f"{viewer_key}_bg_color", defaults["bg_color"]),
        "camera_preset": st.session_state.get(f"{viewer_key}_camera_preset", defaults["camera_preset"]),
        "projection": st.session_state.get(f"{viewer_key}_projection", defaults["projection"]),
        "show_axes": st.session_state.get(f"{viewer_key}_show_axes", defaults["show_axes"]),
    }

    with st.expander("Viewer settings", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            settings["max_faces"] = st.slider(
                "Max faces per mesh",
                min_value=5000,
                max_value=220000,
                value=int(settings["max_faces"]),
                step=5000,
                key=f"{viewer_key}_max_faces",
            )
            settings["camera_preset"] = st.selectbox(
                "Camera",
                ["Isometric", "Front", "Top", "Left"],
                index=["Isometric", "Front", "Top", "Left"].index(settings["camera_preset"]),
                key=f"{viewer_key}_camera_preset",
            )
            settings["projection"] = st.selectbox(
                "Projection",
                ["perspective", "orthographic"],
                index=["perspective", "orthographic"].index(settings["projection"]),
                key=f"{viewer_key}_projection",
            )
            settings["bg_color"] = st.color_picker(
                "Background",
                value=settings["bg_color"],
                key=f"{viewer_key}_bg_color",
            )
        with col_b:
            settings["ambient"] = st.slider("Ambient light", 0.0, 1.0, float(settings["ambient"]), 0.05, key=f"{viewer_key}_ambient")
            settings["diffuse"] = st.slider("Diffuse light", 0.0, 1.0, float(settings["diffuse"]), 0.05, key=f"{viewer_key}_diffuse")
            settings["specular"] = st.slider("Specular", 0.0, 1.0, float(settings["specular"]), 0.05, key=f"{viewer_key}_specular")
            settings["roughness"] = st.slider("Roughness", 0.0, 1.0, float(settings["roughness"]), 0.05, key=f"{viewer_key}_roughness")

            settings["flatshading"] = st.checkbox("Flat shading", value=bool(settings["flatshading"]), key=f"{viewer_key}_flatshading")
            settings["normalize"] = st.checkbox("Normalize scale", value=bool(settings["normalize"]), key=f"{viewer_key}_normalize")
            settings["show_axes"] = st.checkbox("Show axes", value=bool(settings["show_axes"]), key=f"{viewer_key}_show_axes")
            settings["wireframe"] = st.checkbox("Wireframe overlay", value=bool(settings["wireframe"]), key=f"{viewer_key}_wireframe")
            if settings["wireframe"]:
                settings["wireframe_limit"] = st.slider(
                    "Wireframe edge limit",
                    min_value=500,
                    max_value=50000,
                    value=int(settings["wireframe_limit"]),
                    step=500,
                    key=f"{viewer_key}_wireframe_limit",
                )

    try:
        figure = build_plotly_figure(glb_path, settings=settings)
    except Exception as exc:
        show_preview_mode_badge("Static")
        st.warning(f"3D preview failed to load interactively: {exc}")
        try:
            snapshot_bytes = build_static_snapshot(glb_path, str(exc))
            st.image(snapshot_bytes, caption="Static snapshot fallback", use_container_width=True)
        except Exception as snapshot_exc:
            st.info(f"Static snapshot also failed: {snapshot_exc}")
        return

    show_preview_mode_badge("Interactive")
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
        key=chart_key,
    )


def persist_glb_in_session(glb_path):
    """Store GLB bytes in session so preview survives temp-file path loss."""
    if not glb_path or not os.path.exists(glb_path):
        return
    with open(glb_path, "rb") as f:
        glb_bytes = f.read()
    st.session_state["glb_path"] = glb_path
    st.session_state["glb_bytes"] = glb_bytes
    st.session_state["glb_name"] = os.path.basename(glb_path)
    st.session_state.pop("conversion_obj_bytes", None)
    st.session_state.pop("conversion_stl_bytes", None)
    st.session_state.pop("conversion_source_name", None)
    update_snapshot_preview(glb_path)


def load_glb_as_mesh(glb_path):
    """Load a GLB and normalize it to a single mesh for format export."""
    loaded = trimesh.load(glb_path, force="scene")

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        scene_meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not scene_meshes:
            raise ValueError("No mesh geometry found in GLB file")
        mesh = trimesh.util.concatenate(scene_meshes) if len(scene_meshes) > 1 else scene_meshes[0].copy()
    else:
        raise ValueError("Unsupported GLB content")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("GLB mesh has no faces to export")

    return mesh


def export_mesh_bytes(mesh, file_type):
    """Export a trimesh mesh to bytes for Streamlit download buttons."""
    payload = mesh.export(file_type=file_type)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, bytes):
        return payload
    raise ValueError(f"Unsupported export payload for {file_type}")


def get_session_glb_path():
    """Return a valid on-disk GLB path from session, recreating temp file when needed."""
    glb_path = st.session_state.get("glb_path")
    if glb_path and os.path.exists(glb_path):
        st.session_state["glb_restored_from_bytes"] = False
        return glb_path

    glb_bytes = st.session_state.get("glb_bytes")
    if not glb_bytes:
        return ""

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    tmp.write(glb_bytes)
    tmp.close()
    st.session_state["glb_path"] = tmp.name
    st.session_state["glb_restored_from_bytes"] = True
    if "glb_name" not in st.session_state:
        st.session_state["glb_name"] = os.path.basename(tmp.name)
    return tmp.name


def read_binary_file(path):
    """Return file bytes for an existing file path."""
    if not path or not os.path.exists(path):
        return b""
    with open(path, "rb") as f:
        return f.read()


def _subclip(clip, start_time, end_time):
    """Return a trimmed subclip, compatible with MoviePy v1 and v2 APIs."""
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start_time, end_time)
    return clip.subclip(start_time, end_time)


def _apply_fades(clip, fade_in_sec=0.0, fade_out_sec=0.0):
    """Apply fade-in/fade-out effects to a clip, compatible across MoviePy versions."""
    fade_in_sec = float(fade_in_sec or 0)
    fade_out_sec = float(fade_out_sec or 0)
    if fade_in_sec <= 0 and fade_out_sec <= 0:
        return clip

    if hasattr(clip, "with_effects"):
        from moviepy import vfx

        effects = []
        if fade_in_sec > 0:
            effects.append(vfx.FadeIn(fade_in_sec))
        if fade_out_sec > 0:
            effects.append(vfx.FadeOut(fade_out_sec))
        return clip.with_effects(effects)

    if fade_in_sec > 0:
        from moviepy.video.fx import fadein

        clip = clip.fx(fadein, fade_in_sec)
    if fade_out_sec > 0:
        from moviepy.video.fx import fadeout

        clip = clip.fx(fadeout, fade_out_sec)
    return clip


def _apply_video_effects(clip, black_and_white=False, mirror_horizontal=False, mute_audio=False):
    """Apply optional single-clip effects, compatible with MoviePy v1 and v2 APIs."""
    if black_and_white:
        if hasattr(clip, "with_effects"):
            from moviepy import vfx

            clip = clip.with_effects([vfx.BlackAndWhite()])
        else:
            from moviepy.video.fx.blackwhite import blackwhite

            clip = clip.fx(blackwhite)
    if mirror_horizontal:
        if hasattr(clip, "with_effects"):
            from moviepy import vfx

            clip = clip.with_effects([vfx.MirrorX()])
        else:
            from moviepy.video.fx.mirror_x import mirror_x

            clip = clip.fx(mirror_x)
    if mute_audio:
        clip = clip.without_audio() if hasattr(clip, "without_audio") else clip.set_audio(None)
    return clip


def edit_video_with_moviepy(
    input_path,
    output_path,
    start_time,
    end_time,
    speed,
    fade_in_sec=0.0,
    fade_out_sec=0.0,
    black_and_white=False,
    mirror_horizontal=False,
    mute_audio=False,
):
        """Trim, optionally change speed, and apply effects with MoviePy."""
        from moviepy import VideoFileClip

        clip = VideoFileClip(input_path)
        edited_clip = None
        try:
            edited_clip = _subclip(clip, start_time, end_time)

            if speed != 1.0:
                if hasattr(edited_clip, "with_speed_scaled"):
                    edited_clip = edited_clip.with_speed_scaled(factor=speed)
                else:
                    from moviepy.video.fx import speedx

                    edited_clip = edited_clip.fx(speedx, factor=speed)

            edited_clip = _apply_fades(edited_clip, fade_in_sec, fade_out_sec)
            edited_clip = _apply_video_effects(
                edited_clip, black_and_white, mirror_horizontal, mute_audio
            )

            edited_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
        finally:
            if edited_clip is not None:
                edited_clip.close()
            clip.close()


def normalize_hashtags(raw_value):
    """Normalize comma/space separated tags into unique hashtag tokens."""
    tags = []
    seen = set()
    for token in re.split(r"[\s,]+", (raw_value or "").strip()):
        cleaned = re.sub(r"[^\w-]+", "", token).lstrip("#")
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        tags.append(f"#{cleaned}")
    return " ".join(tags)


def extract_primary_suggestion(text):
    """Extract the most actionable suggestion from assistant text."""
    if not text:
        return ""

    fenced_match = re.search(r"```(?:[\w+-]*)\n(.*?)```", text, flags=re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        if candidate:
            return candidate

    labelled_match = re.search(
        r"(?:Suggested prompt|Prompt|Recommendation)\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if labelled_match:
        candidate = labelled_match.group(1).strip()
        if candidate:
            return candidate

    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•").strip()
        if cleaned:
            return cleaned
    return ""


def render_copy_button(label, text, key, disabled=False):
    """Render a browser-side clipboard button without rerunning the app."""
    escaped_label = html.escape(label, quote=True)
    serialized_text = json.dumps(text or "").replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    disabled_attribute = " disabled" if disabled else ""
    components.html(
        f"""
        <button id="{html.escape(key, quote=True)}"{disabled_attribute}
                style="width:100%; min-height:2.4rem; padding:0.25rem 0.75rem;
                       border:1px solid #c8cdd4; border-radius:0.5rem;
                       background:#fff; color:#262730; cursor:pointer; font-size:0.875rem;">
            {escaped_label}
        </button>
        <script>
        const button = document.getElementById({json.dumps(key)});
        const value = {serialized_text};
        button.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(value);
                button.textContent = "Copied!";
                setTimeout(() => button.textContent = {json.dumps(label)}, 1500);
            }} catch (error) {{
                const area = document.createElement("textarea");
                area.value = value;
                document.body.appendChild(area);
                area.select();
                document.execCommand("copy");
                area.remove();
                button.textContent = "Copied!";
                setTimeout(() => button.textContent = {json.dumps(label)}, 1500);
            }}
        }});
        </script>
        """,
        height=44,
    )


def build_assistant_user_prompt(user_input):
    """Build a simple prompt so the assistant can recommend creative ideas."""
    return (
        "You are a helpful creative assistant for a Streamlit app that goes from text prompt → image → 3D model → export.\n"
        "Recommend fresh image prompts, styling ideas, and 3D model concepts that work well for 3D-printable figurines.\n"
        "Keep responses concise and practical. Use short bullets and include one clearly marked 'Suggested prompt:' line when relevant.\n\n"
        "User request:\n"
        + user_input.strip()
    )


def setup_pwa():
    """Setup PWA manifest and service worker."""
    pwa_html = """
    <script>
    const safeDocument = () => {
        try {
            if (window.parent && window.parent !== window && window.parent.document) {
                return window.parent.document;
            }
        } catch (err) {
            // parent is inaccessible due to cross-origin or sandbox restrictions
        }
        return document;
    };

    const ensureHeadTag = (doc, tagName, attrs) => {
        const selector = tagName + Object.entries(attrs)
            .map(([key, value]) => `[${key}="${value}"]`)
            .join('');

        if (!doc.querySelector(selector)) {
            const element = doc.createElement(tagName);
            Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
            doc.head.appendChild(element);
        }
    };

    const attachPwaMeta = () => {
        const doc = safeDocument();
        ensureHeadTag(doc, 'link', { rel: 'manifest', href: '/manifest.json' });
        ensureHeadTag(doc, 'meta', { name: 'theme-color', content: '#ff4b4b' });
        ensureHeadTag(doc, 'meta', { name: 'apple-mobile-web-app-capable', content: 'yes' });
        ensureHeadTag(doc, 'meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'default' });
        ensureHeadTag(doc, 'meta', { name: 'apple-mobile-web-app-title', content: '3D Generator' });
        ensureHeadTag(doc, 'link', {
            rel: 'apple-touch-icon',
            href: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTkyIiBoZWlnaHQ9IjE5MiIgdmlld0JveD0iMCAwIDE5MiAxOTIiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxOTIiIGhlaWdodD0iMTkyIiByeD0iMjQiIGZpbGw9IiNmZjRiNGIiLz4KPHBhdGggZD0iTTEyMCA2NGgzMnY2NGgtMzJ2LTY0ek0xMjggOTZ2MzJoLTE2di0zMmgxNnoiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo='
        });
    };

    const registerServiceWorker = () => {
        const targetWindow = (window.parent && window.parent !== window ? window.parent : window);
        if (targetWindow.navigator && 'serviceWorker' in targetWindow.navigator) {
            targetWindow.addEventListener('load', () => {
                targetWindow.navigator.serviceWorker.register('/sw.js', { updateViaCache: 'none' })
                    .then((registration) => {
                        registration.update();
                        console.log('ServiceWorker registration successful');
                    })
                    .catch(err => console.log('ServiceWorker registration failed:', err));
            });
        }
    };

    let deferredPrompt;
    const setupInstallPrompt = () => {
        const targetWindow = (window.parent && window.parent !== window ? window.parent : window);
        targetWindow.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            const installButton = document.getElementById('install-button');
            if (installButton) {
                installButton.style.display = 'block';
            }
        });
    };

    const bindInstallButton = () => {
        const installButton = document.getElementById('install-button');
        if (installButton) {
            installButton.addEventListener('click', async () => {
                if (!deferredPrompt) return;
                deferredPrompt.prompt();
                const { outcome } = await deferredPrompt.userChoice;
                deferredPrompt = null;
                installButton.style.display = 'none';
                console.log('Install prompt outcome:', outcome);
            });
        }
    };

    attachPwaMeta();
    registerServiceWorker();
    setupInstallPrompt();
    bindInstallButton();
    </script>
    """

    st.markdown(pwa_html, unsafe_allow_html=True)

def run_app():
    st.set_page_config(
        page_title="Image → 3D Generator",
        layout="wide",
        page_icon="🧊",
        initial_sidebar_state="collapsed",
    )

    setup_pwa()

    st.title("RKstudio3Dps & Ecommerce App → 🖼️ → 🧊")

    install_button_html = """
<div style="position: fixed; top: 10px; right: 10px; z-index: 1000;">
    <button id="install-button" style="
        background: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        display: none;
    ">
        📱 Install App
    </button>
</div>

<script>
const installButton = document.getElementById('install-button');

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    window.deferredPrompt = e;
    installButton.style.display = 'block';
});

installButton.addEventListener('click', async () => {
    const promptEvent = window.deferredPrompt;
    if (!promptEvent) return;

    promptEvent.prompt();
    const { outcome } = await promptEvent.userChoice;

    window.deferredPrompt = null;
    installButton.style.display = 'none';
});
</script>
"""

    st.markdown(install_button_html, unsafe_allow_html=True)

    tab_details, tab_prompt, tab_chat, tab1, tab2, tab3, tab4_obj, tab4_stl, tab_blog, tab_video, tab_social, tab_links = st.tabs([
        "Details",
        "3D Model Prompt Ideas (Cohere)",
        "AI Chat Assistant",
        "Image Generation",
        "3D Model Generation",
        "3D File Viewer",
        "Convert to OBJ",
        "Convert to STL",
        "Blog Writer (Cohere)",
        "Video Editor (MoviePy)",
        "Social Share",
        "Links",
    ])

    def sanitize_filename(name):
        return re.sub(r"[^\w\-]+", "_", (name or "").strip())

    with tab1:
        st.caption("Remember to check API usage!")
        if "image_prompt_value" not in st.session_state:
            st.session_state["image_prompt_value"] = "One Single Stylized simple multicoloured Common Holly performing Heterophylly whilst presented on a sturdy figurine base suitable for 3D printing."
        mode = "Fast_Mode_OA"
        image_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            key="image_openai_api_key",
            help="Used only for this browser session and not saved to this app's configuration.",
        )
        prompt = st.text_area(
            "Image prompt",
            value=st.session_state["image_prompt_value"],
            key="image_prompt",
        )

        if st.button("Generate Image"):
            with st.spinner("Generating image..."):
                image_path = (
                    generate_image_openai(prompt, image_api_key)
                )
                if image_path:
                    st.session_state["image_path"] = image_path
                    st.session_state["last_image_generation_status"] = "Image generated successfully."
                    st.session_state["last_generated_image_prompt"] = prompt
                    st.image(image_path, caption="Generated Image", width=400)
                else:
                    st.session_state["last_image_generation_status"] = "Image generation failed."

    with tab2:
        st.caption("Remember to check API usage!")
        mode = "Medium_Quality_Mode_STA"
        model_api_key = st.text_input(
            "Stability AI API key",
            type="password",
            key="model_stability_api_key",
            help="Used only for this browser session and not saved to this app's configuration.",
        )
        image_path = st.session_state.get("image_path")

        uploaded = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], key="model_image_upload")
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(uploaded.read())
            tmp.close()
            image_path = tmp.name

        if not image_path:
            st.warning("Please generate or upload an image first.")
        else:
            st.image(image_path, caption="Input Image", width=300)

            if mode in ["Medium_Quality_Mode_STA", "Medium_Quality_Mode"]:
                texture_resolution = st.selectbox("Texture Resolution", ["512", "1024", "2048"], index=1)
                foreground_ratio = st.slider("Foreground Ratio", 0.1, 1.0, 0.85, 0.05)
                remesh = st.selectbox("Remesh", ["none", "quad", "triangle"])
                vertex_count = st.number_input("Vertex Count (-1 = auto)", value=-1)
                filename_glb = st.text_input("GLB filename (no extension)", value="model", key="filename_glb_tab2_sta")

                if st.button("Generate 3D Model", key="generate_3d_stability"):
                    with st.spinner("Generating 3D model..."):
                        glb_path = generate_3d_model_stability(
                            image_path,
                            texture_resolution,
                            foreground_ratio,
                            remesh,
                            vertex_count,
                            model_api_key,
                        )
                        if glb_path:
                            persist_glb_in_session(glb_path)
                            st.session_state["last_model_generation_status"] = "3D model generated successfully."
                            st.success("3D model generated!")
                            safe_name = sanitize_filename(filename_glb) or "model"
                            glb_download_bytes = read_binary_file(glb_path)
                            st.download_button(
                                "Download GLB",
                                glb_download_bytes,
                                file_name=f"{safe_name}.glb",
                                mime="model/gltf-binary",
                            )
                        else:
                            st.session_state["last_model_generation_status"] = "3D model generation failed."
            current_glb_path = get_session_glb_path()
            if current_glb_path and os.path.exists(current_glb_path):
                snapshot_png = st.session_state.get("glb_preview_png")
                if snapshot_png:
                    st.subheader("Model Image Preview")
                    st.image(snapshot_png, caption="Snapshot of generated GLB", use_container_width=True)
                st.subheader("Current 3D Preview")
                restored_tag = "restored-from-session" if st.session_state.get("glb_restored_from_bytes") else "direct-file"
                st.caption(f"Debug render source: {current_glb_path} ({restored_tag})")
                display_3d_model(current_glb_path, chart_key="tab2_plotly_preview")
            elif st.session_state.get("glb_bytes"):
                st.info("GLB exists in session but could not be restored to disk for preview.")

    with tab3:
        st.caption("Preview an existing GLB file or inspect the most recently generated model.")
        viewer_upload = st.file_uploader("Upload a GLB file", type=["glb"], key="viewer_glb_upload")

        if viewer_upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            tmp.write(viewer_upload.read())
            tmp.close()
            persist_glb_in_session(tmp.name)

        current_glb_path = get_session_glb_path()
        if current_glb_path and os.path.exists(current_glb_path):
            snapshot_png = st.session_state.get("glb_preview_png")
            if snapshot_png:
                st.subheader("Model Image Preview")
                st.image(snapshot_png, caption="Snapshot of selected GLB", use_container_width=True)
            restored_tag = "restored-from-session" if st.session_state.get("glb_restored_from_bytes") else "direct-file"
            st.caption(f"Debug render source: {current_glb_path} ({restored_tag})")
            display_3d_model(current_glb_path, chart_key="tab3_plotly_preview")
            default_glb_name = os.path.splitext(st.session_state.get("glb_name", os.path.basename(current_glb_path)))[0]
            filename_glb_view = st.text_input("GLB filename (no extension)", value=default_glb_name, key="filename_glb_tab3")
            safe_name = sanitize_filename(filename_glb_view) or "model"
            glb_view_bytes = read_binary_file(current_glb_path)
            st.download_button(
                "Download Current GLB",
                glb_view_bytes,
                file_name=f"{safe_name}.glb",
                mime="model/gltf-binary",
                key="download_current_glb",
            )
        else:
            st.info("Upload a .glb file here or generate one in the 3D Model Generation tab.")

    with tab4_obj:
        st.caption("Convert a GLB to OBJ and choose the output filename.")
        obj_upload = st.file_uploader("Upload a GLB for OBJ conversion (optional)", type=["glb"], key="obj_conversion_glb_upload")

        if obj_upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            tmp.write(obj_upload.read())
            tmp.close()
            st.session_state["obj_glb_path"] = tmp.name

        obj_glb_path = st.session_state.get("obj_glb_path") or get_session_glb_path()
        if obj_glb_path and os.path.exists(obj_glb_path):
            source_name = os.path.splitext(os.path.basename(obj_glb_path))[0]
            st.write(f"Current GLB source: {os.path.basename(obj_glb_path)}")
            filename_obj = st.text_input("OBJ filename (no extension)", value=source_name, key="filename_obj_tab")

            if st.button("Convert to OBJ", key="convert_to_obj"):
                try:
                    mesh = load_glb_as_mesh(obj_glb_path)
                    st.session_state["obj_bytes"] = export_mesh_bytes(mesh, "obj")
                    st.session_state["last_conversion_status"] = "OBJ conversion complete."
                    st.success("Conversion to OBJ complete.")
                except Exception as exc:
                    st.session_state["last_conversion_status"] = f"OBJ conversion failed: {exc}"
                    st.error(f"OBJ conversion failed: {exc}")

            obj_bytes = st.session_state.get("obj_bytes")
            if obj_bytes:
                safe_name_obj = sanitize_filename(filename_obj) or "model"
                st.download_button(
                    "Download OBJ",
                    obj_bytes,
                    file_name=f"{safe_name_obj}.obj",
                    mime="model/obj",
                    key="download_obj",
                )
        else:
            st.info("Upload a .glb file here, or generate one in the 3D Model Generation tab.")

    with tab4_stl:
        st.caption("Convert a GLB to STL and choose the output filename.")
        stl_upload = st.file_uploader("Upload a GLB for STL conversion (optional)", type=["glb"], key="stl_conversion_glb_upload")

        if stl_upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            tmp.write(stl_upload.read())
            tmp.close()
            st.session_state["stl_glb_path"] = tmp.name

        stl_glb_path = st.session_state.get("stl_glb_path") or get_session_glb_path()
        if stl_glb_path and os.path.exists(stl_glb_path):
            source_name = os.path.splitext(os.path.basename(stl_glb_path))[0]
            st.write(f"Current GLB source: {os.path.basename(stl_glb_path)}")
            filename_stl = st.text_input("STL filename (no extension)", value=source_name, key="filename_stl_tab")

            if st.button("Convert to STL", key="convert_to_stl"):
                try:
                    mesh = load_glb_as_mesh(stl_glb_path)
                    st.session_state["stl_bytes"] = export_mesh_bytes(mesh, "stl")
                    st.session_state["last_conversion_status"] = "STL conversion complete."
                    st.success("Conversion to STL complete.")
                except Exception as exc:
                    st.session_state["last_conversion_status"] = f"STL conversion failed: {exc}"
                    st.error(f"STL conversion failed: {exc}")

            stl_bytes = st.session_state.get("stl_bytes")
            if stl_bytes:
                safe_name_stl = sanitize_filename(filename_stl) or "model"
                st.download_button(
                    "Download STL",
                    stl_bytes,
                    file_name=f"{safe_name_stl}.stl",
                    mime="model/stl",
                    key="download_stl",
                )
        else:
            st.info("Upload a .glb file here, or generate one in the 3D Model Generation tab.")

    with tab_chat:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.15rem;">
                <span style="display:inline-flex;align-items:center;justify-content:center;width:1.25rem;height:1.25rem;border-radius:0.35rem;background:#e0f2fe;border:1px solid #bae6fd;overflow:hidden;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <rect x="4" y="7" width="16" height="12" rx="3" stroke="#0369a1" stroke-width="1.7"/>
                        <circle cx="9" cy="13" r="1.4" fill="#0284c7"/>
                        <circle cx="15" cy="13" r="1.4" fill="#0284c7"/>
                        <rect x="10" y="16.2" width="4" height="1.6" rx="0.8" fill="#0284c7"/>
                        <path d="M12 3.8V7" stroke="#0369a1" stroke-width="1.7" stroke-linecap="round"/>
                        <circle cx="12" cy="3" r="1.2" fill="#0369a1"/>
                    </svg>
                </span>
                <span style="font-size:0.92rem;color:#475569;">Ask the assistant for image prompts, styling ideas, and model concepts.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_cohere_api_key_input("cohere_api_key_chat")
        cohere_api_key = get_cohere_api_key()
        if cohere_api_key:
            st.success("Cohere key detected. Assistant is ready.")
        else:
            st.warning("Enter a Cohere API key above, or configure one in .env or Streamlit secrets.")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "chat_input" not in st.session_state:
            st.session_state["chat_input"] = ""
        if "chat_input_next" not in st.session_state:
            st.session_state["chat_input_next"] = None
        if "agent_last_status" not in st.session_state:
            st.session_state["agent_last_status"] = ""
        if "agent_last_suggestion" not in st.session_state:
            st.session_state["agent_last_suggestion"] = ""
        if "cohere_model" not in st.session_state:
            st.session_state["cohere_model"] = "command-r-plus-08-2024"
        if "cohere_temperature" not in st.session_state:
            st.session_state["cohere_temperature"] = 0.7
        if "cohere_max_tokens" not in st.session_state:
            st.session_state["cohere_max_tokens"] = 450

        # Apply pending chat input updates before the widget is instantiated.
        if st.session_state["chat_input_next"] is not None:
            st.session_state["chat_input"] = st.session_state["chat_input_next"]
            st.session_state["chat_input_next"] = None

        with st.expander("Chat Settings", expanded=True):
            st.selectbox(
                "Cohere model",
                [
                    "command-r-plus-08-2024",
                    "command-a-03-2025",
                    "command-r7b-12-2024",
                ],
                key="cohere_model",
            )
            st.slider("Temperature", 0.0, 1.0, key="cohere_temperature", step=0.05)
            st.slider("Max output tokens", 100, 2000, key="cohere_max_tokens", step=50)

        cols = st.columns(3)
        with cols[0]:
            if st.button("Use starter question", key="starter_question"):
                st.session_state["chat_input_next"] = "Give me 5 creative image ideas that convert well into printable 3D figurines."
                st.rerun()
        with cols[1]:
            if st.button("Clear chat messages", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.session_state["chat_input_next"] = ""
                st.session_state["agent_last_status"] = "Chat messages cleared (settings kept)."
                st.rerun()
        with cols[2]:
            if st.button("Reset chat", key="reset_assistant_state"):
                st.session_state["chat_history"] = []
                st.session_state["chat_input_next"] = ""
                st.session_state["agent_last_suggestion"] = ""
                st.session_state["agent_last_status"] = "Assistant state reset."
                st.rerun()

        if st.session_state.get("agent_last_status"):
            st.info(st.session_state.get("agent_last_status"))

        user_input = st.text_input(
            "Ask the assistant for ideas",
            key="chat_input",
            placeholder="Example: Give me creative image ideas that convert well into printable 3D figurines.",
        )
        if st.button("Send", key="send_chat"):
            if not user_input.strip():
                st.info("Type a prompt first.")
            elif not cohere_api_key:
                st.error("Enter a Cohere API key above to use the assistant.")
            else:
                try:
                    history = st.session_state["chat_history"]
                    selected_model = st.session_state.get("cohere_model", "command-a-03-2025")
                    temperature = st.session_state.get("cohere_temperature", 0.7)
                    max_tokens = st.session_state.get("cohere_max_tokens", 450)
                    grounded_input = build_assistant_user_prompt(user_input)
                    model_candidates = [
                        selected_model,
                        "command-r-plus-08-2024",
                        "command-a-03-2025",
                        "command-r7b-12-2024",
                    ]
                    # Preserve order while deduplicating.
                    model_candidates = list(dict.fromkeys(model_candidates))

                    last_error = None
                    answer = ""
                    for model_name in model_candidates:
                        try:
                            with st.spinner("Assistant is thinking..."):
                                answer = get_cohere_response(
                                    api_key=cohere_api_key,
                                    user_input=grounded_input,
                                    history=history,
                                    model_name=model_name,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                )
                            if answer:
                                st.session_state["cohere_model"] = model_name
                                st.session_state["agent_last_status"] = (
                                    f"Response generated with {model_name}."
                                )
                                break
                        except Exception as exc:
                            last_error = exc

                    if not answer and last_error:
                        raise last_error

                    if not answer:
                        answer = "I could not generate a response. Please try again."
                        st.session_state["agent_last_status"] = "Model returned an empty response; fallback text was used."

                    suggestion = extract_primary_suggestion(answer)
                    st.session_state["chat_history"].append(
                        {"id": f"user-{uuid.uuid4().hex}", "role": "user", "content": user_input}
                    )
                    st.session_state["chat_history"].append(
                        {
                            "id": f"assistant-{uuid.uuid4().hex}",
                            "role": "assistant",
                            "content": answer,
                            "suggestion": suggestion,
                        }
                    )
                    st.session_state["agent_last_suggestion"] = suggestion
                    st.session_state["chat_input_next"] = ""
                    st.rerun()
                except Exception as exc:
                    st.session_state["agent_last_status"] = f"Assistant request failed: {exc}"
                    st.error(f"Cohere chat failed: {exc}")

        for idx, msg in enumerate(st.session_state["chat_history"]):
            if "id" not in msg:
                msg["id"] = f"legacy-{uuid.uuid4().hex}"
            message_key = msg["id"]
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    if "suggestion" not in msg:
                        suggestion = extract_primary_suggestion(msg["content"])
                        msg["suggestion"] = suggestion
                    suggestion = msg.get("suggestion", "")
                    action_cols = st.columns(5)
                    with action_cols[0]:
                        render_copy_button(
                            "Copy to clipboard",
                            suggestion,
                            f"copy_suggestion_{message_key}",
                            disabled=not bool(suggestion),
                        )
                    with action_cols[1]:
                        if st.button(
                            "Apply suggested prompt",
                            key=f"apply_suggested_prompt_{message_key}",
                            disabled=not bool(suggestion),
                        ):
                            st.session_state["prompt_topic"] = suggestion
                            st.session_state["agent_last_status"] = "Applied suggestion to 3D prompt ideas topic."
                            st.rerun()
                    with action_cols[2]:
                        if st.button(
                            "Use as image prompt",
                            key=f"copy_image_prompt_{message_key}",
                            disabled=not bool(suggestion),
                        ):
                            st.session_state["image_prompt_value"] = suggestion
                            st.session_state["agent_last_status"] = "Copied suggestion to image prompt field."
                            st.rerun()
                    with action_cols[3]:
                        if st.button(
                            "Use as blog topic",
                            key=f"copy_blog_topic_{message_key}",
                            disabled=not bool(suggestion),
                        ):
                            st.session_state["blog_question"] = suggestion
                            st.session_state["agent_last_status"] = "Copied suggestion to blog draft topic."
                            st.rerun()
                    with action_cols[4]:
                        if st.button(
                            "Prepare social caption",
                            key=f"prepare_social_caption_{message_key}",
                            disabled=not bool(suggestion),
                        ):
                            st.session_state["social_post_caption"] = suggestion
                            st.session_state["social_post_tags"] = st.session_state.get("social_post_tags", DEFAULT_SOCIAL_TAGS)
                            st.session_state["social_post_cta"] = st.session_state.get("social_post_cta", DEFAULT_SOCIAL_CTA)
                            st.session_state["agent_last_status"] = "Prepared social caption fields from assistant suggestion."
                            st.rerun()

    with tab_blog:
        st.subheader("Cohere Blog Writer")
        st.caption("Ask Cohere a question and generate a structured blog draft in Markdown.")

        render_cohere_api_key_input("cohere_api_key_blog")
        cohere_api_key = get_cohere_api_key()
        if cohere_api_key:
            st.success("Cohere key detected. Blog writer is ready.")
        else:
            st.warning("Enter a Cohere API key above, or configure one in .env or Streamlit secrets.")

        if "blog_question" not in st.session_state:
            st.session_state["blog_question"] = "How can small studios use AI to speed up 3D concept workflows?"
        if "blog_audience" not in st.session_state:
            st.session_state["blog_audience"] = "Indie creators and hobby makers"
        if "blog_tone" not in st.session_state:
            st.session_state["blog_tone"] = "Practical and friendly"
        if "blog_length" not in st.session_state:
            st.session_state["blog_length"] = 900
        if "blog_include_outline" not in st.session_state:
            st.session_state["blog_include_outline"] = True
        if "blog_output" not in st.session_state:
            st.session_state["blog_output"] = ""
        if "blog_model" not in st.session_state:
            st.session_state["blog_model"] = "command-r-plus-08-2024"

        st.selectbox(
            "Cohere model",
            [
                "command-r-plus-08-2024",
                "command-a-03-2025",
                "command-r7b-12-2024",
            ],
            key="blog_model",
        )
        st.text_area(
            "Question or topic for the blog",
            key="blog_question",
            placeholder="Example: What are the best ways to create 3D-printable character concepts from text prompts?",
            height=100,
        )
        st.text_input("Target audience", key="blog_audience")
        st.text_input("Tone", key="blog_tone")
        st.slider("Target word count", min_value=300, max_value=2000, step=100, key="blog_length")
        st.checkbox("Include a short outline at the top", key="blog_include_outline")

        if st.button("Generate Blog Draft", key="generate_blog_draft"):
            if not st.session_state["blog_question"].strip():
                st.info("Add a question or topic first.")
            elif not cohere_api_key:
                st.error("Enter a Cohere API key above to generate a blog draft.")
            else:
                try:
                    include_outline = "yes" if st.session_state.get("blog_include_outline") else "no"
                    user_prompt = (
                        "You are a senior technical blog writer. Write a clear, engaging Markdown blog post.\\n"
                        f"Primary question/topic: {st.session_state.get('blog_question', '').strip()}\\n"
                        f"Target audience: {st.session_state.get('blog_audience', '').strip()}\\n"
                        f"Tone: {st.session_state.get('blog_tone', '').strip()}\\n"
                        f"Target length: about {int(st.session_state.get('blog_length', 900))} words\\n"
                        f"Include a short outline first: {include_outline}\\n\\n"
                        "Return Markdown only with this structure:\\n"
                        "1) Title\\n"
                        "2) (Optional) Outline\\n"
                        "3) Introduction\\n"
                        "4) Main sections with H2/H3 headings\\n"
                        "5) Conclusion with practical next steps\\n"
                        "Keep the content actionable and include concrete examples where useful."
                    )

                    with st.spinner("Generating blog draft with Cohere..."):
                        blog_text = get_cohere_response(
                            api_key=cohere_api_key,
                            user_input=user_prompt,
                            history=[],
                            model_name=st.session_state.get("blog_model", "command-r-plus-08-2024"),
                            temperature=0.6,
                            max_tokens=1800,
                        )

                    st.session_state["blog_output"] = blog_text
                    st.success("Blog draft generated.")
                except Exception as exc:
                    st.error(f"Blog generation failed: {exc}")

        blog_output = st.session_state.get("blog_output", "")
        if blog_output:
            st.markdown("### Draft Preview")
            st.markdown(blog_output)
            blog_name = sanitize_filename(st.session_state.get("blog_question", "blog-draft")[:60]) or "blog-draft"
            st.download_button(
                "Download Blog Markdown",
                data=blog_output,
                file_name=f"{blog_name}.md",
                mime="text/markdown",
                key="download_blog_markdown",
            )

    with tab_prompt:
        st.subheader("3D Image Prompt Ideas (Cohere)")
        st.caption("Enter a topic and a visual style to generate 3D image prompt ideas.")

        render_cohere_api_key_input("cohere_api_key_prompt")
        cohere_api_key = get_cohere_api_key()
        if cohere_api_key:
            st.success("Cohere key detected. Prompt ideas are ready.")
        else:
            st.warning("Enter a Cohere API key above, or configure one in .env or Streamlit secrets.")

        _STYLE_PRESETS = {
            "As Simple As Possible": (
                "extremely simple and minimal style optimised for clean 3D model conversion",
                "single object centered on a plain white background, basic geometric shapes only, no fine detail, no accessories, no background elements, solid flat colors, even neutral lighting, nothing that would confuse a 3D mesh generator",
            ),
            "Simple & Cartoony": (
                "simple, playful cartoon style",
                "rounded shapes, smooth surfaces, toy-like proportions, bright but limited colors, soft even lighting, no complex details",
            ),
            "Realistic & Detailed": (
                "photorealistic and highly detailed style",
                "accurate proportions, fine surface detail, realistic materials and textures, natural lighting with shadows",
            ),
            "Low-poly / Stylized": (
                "low-poly stylized style",
                "flat shaded geometric facets, minimal color palette, clean silhouette, isometric-friendly proportions",
            ),
            "Sci-fi / Futuristic": (
                "sci-fi futuristic style",
                "sleek hard-surface forms, metallic and glowing materials, neon accent colors, dramatic rim lighting",
            ),
            "Hand-crafted / Clay": (
                "hand-crafted clay or stop-motion style",
                "slightly imperfect organic shapes, matte clay-like surfaces, warm studio lighting, tactile texture cues",
            ),
        }

        if "prompt_topic" not in st.session_state:
            st.session_state["prompt_topic"] = ""
        if "prompt_examples_count" not in st.session_state:
            st.session_state["prompt_examples_count"] = 5
        if "prompt_examples_audience" not in st.session_state:
            st.session_state["prompt_examples_audience"] = "3D artists, illustrators, and hobby makers"
        if "prompt_style_preset" not in st.session_state:
            st.session_state["prompt_style_preset"] = "As Simple As Possible"
        if "prompt_style_custom" not in st.session_state:
            st.session_state["prompt_style_custom"] = ""
        if "prompt_examples_output" not in st.session_state:
            st.session_state["prompt_examples_output"] = ""
        if "prompt_examples_model" not in st.session_state:
            st.session_state["prompt_examples_model"] = "command-r-plus-08-2024"

        st.selectbox(
            "Cohere model",
            [
                "command-r-plus-08-2024",
                "command-a-03-2025",
                "command-r7b-12-2024",
            ],
            key="prompt_examples_model",
        )
        st.text_area(
            "Topic for 3D image ideas",
            key="prompt_topic",
            placeholder="Example: a friendly robot baker with rounded shapes and bright colors",
            height=100,
        )
        st.selectbox(
            "Visual style",
            list(_STYLE_PRESETS.keys()) + ["Custom…"],
            key="prompt_style_preset",
        )
        if st.session_state.get("prompt_style_preset") == "Custom…":
            st.text_input(
                "Describe your custom style",
                key="prompt_style_custom",
                placeholder="Example: dark fantasy with mossy stone textures and dramatic shadows",
            )
        st.text_input("Intended use or audience", key="prompt_examples_audience")
        st.slider("Number of examples", min_value=3, max_value=10, step=1, key="prompt_examples_count")

        if st.button("Generate 3D Prompt Ideas", key="generate_prompt_examples"):
            if not st.session_state["prompt_topic"].strip():
                st.info("Add a topic first.")
            elif not cohere_api_key:
                st.error("Enter a Cohere API key above to generate prompt ideas.")
            else:
                try:
                    chosen_preset = st.session_state.get("prompt_style_preset", "Simple & Cartoony")
                    if chosen_preset == "Custom…":
                        custom_style = st.session_state.get("prompt_style_custom", "").strip()
                        if not custom_style:
                            st.info("Describe your custom style above first.")
                            st.stop()
                        style_desc = custom_style
                        style_constraints = custom_style
                        display_name = "Custom"
                    else:
                        style_desc, style_constraints = _STYLE_PRESETS.get(
                            chosen_preset, _STYLE_PRESETS["Simple & Cartoony"]
                        )
                        display_name = chosen_preset
                    user_prompt = (
                        f"You are an expert 3D image prompt engineer. Generate prompt ideas for creating 3D images in a {style_desc}.\n"
                        f"Topic: {st.session_state.get('prompt_topic', '').strip()}\n"
                        f"Intended use or audience: {st.session_state.get('prompt_examples_audience', '').strip()}\n"
                        f"Visual style: {display_name}\n"
                        f"Number of examples: {int(st.session_state.get('prompt_examples_count', 5))}\n\n"
                        "Return Markdown only with this structure:\n"
                        "1) A short title for each example\n"
                        "2) A detailed prompt inside a fenced code block\n"
                        "3) One brief sentence explaining what kind of 3D image it is best for\n\n"
                        f"Tailor every prompt to the chosen style. Key visual cues to include: {style_constraints}."
                    )

                    with st.spinner(f"Generating {display_name} 3D prompt ideas with Cohere..."):
                        prompt_examples_text = get_cohere_response(
                            api_key=cohere_api_key,
                            user_input=user_prompt,
                            history=[],
                            model_name=st.session_state.get("prompt_examples_model", "command-r-plus-08-2024"),
                            temperature=0.75,
                            max_tokens=1800,
                        )

                    st.session_state["prompt_examples_output"] = prompt_examples_text
                    st.success(f"{display_name} 3D prompt ideas generated.")
                except Exception as exc:
                    st.error(f"3D prompt generation failed: {exc}")

        prompt_examples_output = st.session_state.get("prompt_examples_output", "")
        if prompt_examples_output:
            chosen_preset = st.session_state.get("prompt_style_preset", "Simple & Cartoony")
            display_name = "Custom" if chosen_preset == "Custom…" else chosen_preset
            st.markdown(f"### Generated {display_name} 3D Prompt Ideas")
            st.markdown(prompt_examples_output)
            prompt_name = sanitize_filename(st.session_state.get("prompt_topic", "prompt-ideas")[:60]) or "prompt-ideas"
            st.download_button(
                "Download Prompt Ideas Markdown",
                data=prompt_examples_output,
                file_name=f"{prompt_name}.md",
                mime="text/markdown",
                key="download_prompt_examples_markdown",
            )

    with tab_details:
        st.subheader("Details")
        st.markdown("""
- **3D Print Art** — Cults3D, Tinkercad
- **Ecommerce** — Shopify, TikTok Shop, Printful, Etsy
- **Social Editing** — Canva, TikTok
- **Video Editing** — MoviePy
- **Social Sharing** — export generated images or 3D previews for posting
- **Blog** — Medium
- **AI** — ChatGPT, Claude
""")

    with tab_video:
        st.subheader("Video Editor (MoviePy)")
        st.caption(
            "Trim, change playback speed, add fades, and apply simple effects before sharing. "
            "Processing runs only when you click Edit Video, so it does not consume resources "
            "while you browse the tab."
        )
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="moviepy_video_upload",
        )
        if uploaded_video:
            suffix = os.path.splitext(uploaded_video.name)[1].lower() or ".mp4"
            source_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            source_file.write(uploaded_video.getvalue())
            source_file.close()
            try:
                from moviepy import VideoFileClip

                source_clip = VideoFileClip(source_file.name)
                duration = max(float(source_clip.duration or 0), 0.1)
                source_clip.close()
            except Exception as exc:
                st.error(f"MoviePy could not read this video: {exc}")
            else:
                st.video(uploaded_video.getvalue())
                st.caption(f"Duration: {duration:.1f} seconds")
                edit_cols = st.columns(3)
                with edit_cols[0]:
                    start_time = st.number_input(
                        "Start (seconds)", min_value=0.0, max_value=duration, value=0.0, step=0.5,
                        key="moviepy_start_time",
                    )
                with edit_cols[1]:
                    end_time = st.number_input(
                        "End (seconds)", min_value=0.1, max_value=duration, value=float(duration),
                        step=0.5, key="moviepy_end_time",
                    )
                with edit_cols[2]:
                    speed = st.selectbox(
                        "Playback speed", [0.5, 1.0, 1.5, 2.0], index=1, key="moviepy_speed",
                    )

                trimmed_duration = max(float(end_time) - float(start_time), 0.0)

                with st.expander("Fades", expanded=False):
                    fade_cols = st.columns(2)
                    with fade_cols[0]:
                        fade_in_sec = st.number_input(
                            "Fade in (seconds)", min_value=0.0,
                            max_value=max(trimmed_duration, 0.0), value=0.0, step=0.1,
                            key="moviepy_fade_in",
                        )
                    with fade_cols[1]:
                        fade_out_sec = st.number_input(
                            "Fade out (seconds)", min_value=0.0,
                            max_value=max(trimmed_duration, 0.0), value=0.0, step=0.1,
                            key="moviepy_fade_out",
                        )

                with st.expander("Effects", expanded=False):
                    black_and_white = st.checkbox(
                        "Black and white", key="moviepy_black_and_white"
                    )
                    mirror_horizontal = st.checkbox(
                        "Mirror horizontally", key="moviepy_mirror_horizontal"
                    )
                    mute_audio = st.checkbox("Mute audio", key="moviepy_mute_audio")

                if st.button("Edit Video", type="primary", key="moviepy_edit_video"):
                    if end_time <= start_time:
                        st.error("End time must be greater than start time.")
                    elif fade_in_sec + fade_out_sec > trimmed_duration:
                        st.error("Fade in + fade out durations cannot exceed the trimmed clip length.")
                    else:
                        edited_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        edited_file.close()
                        try:
                            with st.spinner("Editing video with MoviePy..."):
                                edit_video_with_moviepy(
                                    source_file.name, edited_file.name, start_time, end_time, speed,
                                    fade_in_sec=fade_in_sec,
                                    fade_out_sec=fade_out_sec,
                                    black_and_white=black_and_white,
                                    mirror_horizontal=mirror_horizontal,
                                    mute_audio=mute_audio,
                                )
                            st.session_state["edited_video_path"] = edited_file.name
                            st.success("Edited video is ready in the Social Share tab.")
                        except Exception as exc:
                            st.error(f"Video editing failed: {exc}")

        edited_video_path = st.session_state.get("edited_video_path")
        edited_video_bytes = read_binary_file(edited_video_path)
        if edited_video_bytes:
            st.video(edited_video_bytes)
            st.download_button(
                "Download Edited Video",
                data=edited_video_bytes,
                file_name="edited-video.mp4",
                mime="video/mp4",
                key="download_edited_video",
            )

    with tab_social:
        st.subheader("Social Share")
        st.caption("Prepare a generated image or 3D preview for social posting. Direct TikTok posting is not built in yet.")

        image_path = st.session_state.get("image_path")
        image_bytes = read_binary_file(image_path)
        snapshot_png = st.session_state.get("glb_preview_png")

        asset_options = []
        if image_bytes:
            image_ext = os.path.splitext(image_path)[1].lower() or ".png"
            image_mime = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
            }.get(image_ext, "application/octet-stream")
            asset_options.append(("Generated image", image_bytes, "generated-image", image_ext, image_mime))
        if snapshot_png:
            asset_options.append(("3D model preview", snapshot_png, "3d-model-preview", ".png", "image/png"))
        edited_video_path = st.session_state.get("edited_video_path")
        edited_video_bytes = read_binary_file(edited_video_path)
        if edited_video_bytes:
            asset_options.append(("Edited video", edited_video_bytes, "edited-video", ".mp4", "video/mp4"))

        if not asset_options:
            st.info(
                "Generate an image, a 3D model preview, or an edited video first, then return here "
                "to prepare a social post."
            )
        else:
            asset_labels = [label for label, _, _, _, _ in asset_options]
            selected_label = st.radio("Content to prepare", asset_labels, horizontal=True)
            selected_asset = next((option for option in asset_options if option[0] == selected_label), asset_options[0])
            _, selected_bytes, default_name, default_extension, default_mime = selected_asset

            if default_mime.startswith("video/"):
                st.video(selected_bytes)
            else:
                st.image(selected_bytes, caption=selected_label, use_container_width=True)

            post_format = st.selectbox(
                "Post format",
                list(SOCIAL_POST_FORMAT_INTROS.keys()),
                key="social_post_format",
            )
            filename_social = st.text_input(
                "Download filename (no extension)",
                value=default_name,
                key="filename_social_tab",
            )
            st.text_area(
                "Caption",
                value=st.session_state.get("social_post_caption", DEFAULT_SOCIAL_CAPTION),
                key="social_post_caption",
                height=120,
                placeholder="Describe the model, the style, or the idea behind it.",
            )
            st.text_input(
                "Hashtags (comma or space separated)",
                value=st.session_state.get("social_post_tags", DEFAULT_SOCIAL_TAGS),
                key="social_post_tags",
                placeholder="3d, 3dprinting, characterdesign",
            )
            st.text_input(
                "Call to action",
                value=st.session_state.get("social_post_cta", DEFAULT_SOCIAL_CTA),
                key="social_post_cta",
                placeholder="Example: Follow for more model drops.",
            )

            caption_body = st.session_state.get("social_post_caption", "").strip()
            cta_body = st.session_state.get("social_post_cta", "").strip()
            normalized_tags = normalize_hashtags(st.session_state.get("social_post_tags", DEFAULT_SOCIAL_TAGS))
            has_user_content = any((caption_body, cta_body, normalized_tags))
            formatted_post = ""
            if has_user_content:
                formatted_post = "\n\n".join(
                    part
                    for part in [
                        SOCIAL_POST_FORMAT_INTROS.get(post_format, ""),
                        caption_body,
                        cta_body,
                        normalized_tags,
                    ]
                    if part
                )
            empty_post_message = "Add a caption, CTA, or hashtags to build your post."

            st.markdown("### Caption Preview")
            st.code(formatted_post if has_user_content else empty_post_message, language="markdown")

            safe_name_social = sanitize_filename(filename_social) or default_name
            export_cols = st.columns(2)
            with export_cols[0]:
                st.download_button(
                    "Download Social Asset",
                    data=selected_bytes,
                    file_name=f"{safe_name_social}{default_extension}",
                    mime=default_mime,
                    key="download_social_asset",
                )
            with export_cols[1]:
                st.download_button(
                    "Download Caption Text",
                    data=formatted_post.encode("utf-8"),
                    file_name=f"{safe_name_social}-caption.txt",
                    mime="text/plain",
                    disabled=not bool(formatted_post),
                    key="download_social_caption",
                )

            st.markdown("### Open Platforms")
            social_links = [
                ("TikTok", "https://www.tiktok.com/"),
                ("Canva", "https://www.canva.com/"),
                ("TikTok Shop", "https://shop.tiktok.com/"),
            ]
            social_cols = st.columns(len(social_links))
            for idx, (label, url) in enumerate(social_links):
                with social_cols[idx]:
                    if hasattr(st, "link_button"):
                        st.link_button(label, url, use_container_width=True)
                    else:
                        st.markdown(f"[{label}]({url})")

    with tab_links:
        st.subheader("Links")
        st.caption("Quick links to tools for writing, ecommerce, social sharing, and 3D creation.")

        links = [
            ("Canva", "https://www.canva.com"),
            ("Medium", "https://medium.com"),
            ("ChatGPT", "https://chatgpt.com"),
            ("Claude", "https://claude.ai"),
            ("Shopify", "https://www.shopify.com"),
            ("TikTok Shop", "https://shop.tiktok.com/"),
            ("TikTok", "https://www.tiktok.com/"),
            ("Printful", "https://www.printful.com"),
            ("Etsy", "https://www.etsy.com"),
            ("Tinkercad", "https://www.tinkercad.com"),
            ("Cults3D", "https://cults3d.com"),
        ]

        cols = st.columns(3)
        for idx, (label, url) in enumerate(links):
            with cols[idx % 3]:
                if hasattr(st, "link_button"):
                    st.link_button(label, url, use_container_width=True)
                else:
                    st.markdown(f"[{label}]({url})")

        st.markdown("### API Links")
        st.caption("Direct access to API platforms and docs.")
        api_links = [
            ("Stability API", "https://platform.stability.ai"),
            ("OpenAI API", "https://platform.openai.com"),
            ("Cohere API", "https://docs.cohere.com"),
        ]

        api_cols = st.columns(3)
        for idx, (label, url) in enumerate(api_links):
            with api_cols[idx % 3]:
                if hasattr(st, "link_button"):
                    st.link_button(label, url, use_container_width=True)
                else:
                    st.markdown(f"[{label}]({url})")


def get_openai_client(api_key):
    from openai import OpenAI

    if not api_key:
        raise ValueError("Enter an OpenAI API key to generate an image.")
    return OpenAI(api_key=api_key)


def _friendly_openai_image_error(exc):
    """Map known OpenAI image API failures to user-friendly guidance."""
    text = str(exc)
    lowered = text.lower()

    if "billing_hard_limit_reached" in lowered or "billing hard limit has been reached" in lowered:
        return (
            "OpenAI billing limit reached for the active key/project. "
            "Use a funded key, then enter it above and try again."
        )

    if "invalid_api_key" in lowered or "incorrect api key" in lowered:
        return (
            "OpenAI rejected the API key. Check the key entered above, then retry image generation."
        )

    return f"Image generation failed: {text}"


def generate_image_openai(prompt: str, api_key: str) -> str:
    try:
        client = get_openai_client(api_key)
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )

        image_item = response.data[0]
        image_base64 = getattr(image_item, "b64_json", None)
        image_url = getattr(image_item, "url", None)

        if image_base64:
            image_bytes = base64.b64decode(image_base64)
        elif image_url:
            image_bytes = requests.get(image_url, timeout=60).content
        else:
            raise Exception("OpenAI response did not include image data")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.close()

        return tmp.name
    except Exception as e:
        st.error(_friendly_openai_image_error(e))
        traceback.print_exc()
        return ""


def image_to_3d(host, image_path, stability_key, **kwargs):
    if not stability_key:
        raise ValueError("Enter a Stability AI API key to generate a 3D model.")

    with open(image_path, "rb") as image_file:
        response = requests.post(
            host,
            headers={
                "Authorization": f"Bearer {stability_key}",
                "Accept": "model/gltf-binary",
            },
            files={"image": image_file},
            data=kwargs,
            timeout=180,
        )

    if not response.ok:
        detail = response.text
        if response.status_code == 401:
            raise Exception(f"Stability authentication failed (401). Check STABILITY key. Details: {detail}")
        if response.status_code == 403:
            raise Exception(
                "Stability returned 403 (forbidden). Your API key likely does not have access to the Stable Fast 3D endpoint. "
                f"Details: {detail}"
            )
        raise Exception(f"Stability request failed ({response.status_code}): {detail}")

    return response.content


def generate_3d_model_stability(image_path, texture_resolution, foreground_ratio, remesh, vertex_count, api_key):
    try:
        host = "https://api.stability.ai/v2beta/3d/stable-fast-3d"
        glb_data = image_to_3d(
            host=host,
            image_path=image_path,
            texture_resolution=texture_resolution,
            foreground_ratio=foreground_ratio,
            remesh=remesh,
            vertex_count=vertex_count,
            stability_key=api_key,
        )

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
        tmp.write(glb_data)
        tmp.close()

        return tmp.name
    except Exception as e:
        st.error(f"3D generation failed: {e}")
        traceback.print_exc()
        return ""


if __name__ == "__main__":
    run_app()
