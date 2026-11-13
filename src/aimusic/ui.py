"""Local web UI (Gradio). Start with:  aimusic ui

Runs only on this machine (127.0.0.1). One generation at a time, because the GPU
is the bottleneck and a 4 GB card cannot hold two models.
"""

from __future__ import annotations

import gc
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from .backends import BACKENDS, get_backend
from .pipeline import (
    MAX_DURATION_S,
    build_settings,
    default_backend,
    describe,
    estimate_seconds,
    render,
    save,
    validate,
)

OUTPUT_DIR = Path("output")

EXAMPLE_PROMPTS = [
    "2 min calm soothing piano music with rain for studying",
    "45 sec punk rock with distorted guitars and fast drums",
    "1 min lofi hip hop beats, mellow, 85 bpm",
    "45 sec epic cinematic music",
    "45 sec mellow jazz with soft piano, 70 bpm",
]

BACKEND_HELP = (
    "**acestep**: one continuous piece, up to 10 min (needs the separate ACE-Step install). "
    "**musicgen**: 30 s clips; longer tracks are stitched. **synth**: instant, basic sound, no model."
)


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _preview_md(prompt, duration, tuning, ambience_on, level) -> str:
    """Show how the prompt will be read (instant, no model, no LLM)."""
    if not prompt or not prompt.strip():
        return "*Type a description to see how it will be read.*"
    s = build_settings(
        prompt, duration=duration or None, tuning=tuning or None,
        ambience=bool(ambience_on), ambience_level=level,
    )
    rows = "\n".join(f"- **{label}:** {value}" for label, value in describe(s))
    note = ""
    if s.duration > MAX_DURATION_S:
        note = f"\n\n> Over the {MAX_DURATION_S // 60}-minute limit: shorten it before generating."
    return "**How I read your request**\n\n" + rows + note


def _cleanup_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(prompt, backend_name, duration, seed, tuning, ambience_on, level, lyrics,
             progress=gr.Progress()):
    """Generator: yields (status, audio_path, settings_json, file_path) once a second."""
    if not prompt or not prompt.strip():
        raise gr.Error("Describe the music first.")
    try:
        s = build_settings(
            prompt,
            duration=duration or None,
            seed=None if seed in (None, "") else int(seed),
            lyrics=lyrics or None,
            tuning=tuning or None,
            ambience=bool(ambience_on),
            ambience_level=level,
        )
        warnings = validate(s, backend_name)
    except ValueError as e:
        raise gr.Error(str(e)) from e

    est = estimate_seconds(backend_name, s.duration)
    state: dict = {}

    def work() -> None:
        backend = None
        try:
            backend = get_backend(backend_name)
            state["backend"] = backend
            audio = render(backend, s)
            slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:30] or "track"
            out = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.wav"
            meta = save(audio, backend.sample_rate, out, s, backend.name)
            state["result"] = (out, meta)
        except BaseException as e:  # noqa: BLE001 - reported to the UI below
            state["error"] = e
        finally:
            del backend
            state.pop("backend", None)
            _cleanup_gpu()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    start = time.time()
    warn_text = ("\n\n" + "\n".join(f"> {w}" for w in warnings)) if warnings else ""
    try:
        while thread.is_alive():
            elapsed = time.time() - start
            progress(min(elapsed / est, 0.97),
                     desc=f"Generating... {_fmt_time(elapsed)} elapsed (rough estimate {_fmt_time(est)})")
            yield (f"Generating with **{backend_name}**... {_fmt_time(elapsed)} elapsed. "
                   f"The estimate ({_fmt_time(est)}) is rough and depends on your GPU." + warn_text,
                   None, None, None)
            time.sleep(1)
    finally:
        # If the user pressed Stop (generator closed) while still running, stop the backend.
        if thread.is_alive():
            backend = state.get("backend")
            if backend is not None:
                backend.cancel()
    thread.join()

    if "error" in state:
        raise gr.Error(str(state["error"])[:600])
    out, meta = state["result"]
    took = time.time() - start
    yield (f"Done in {_fmt_time(took)}. Saved to `{out}`." + warn_text,
           str(out.resolve()), s.to_dict(), str(out.resolve()))


def build_app() -> gr.Blocks:
    with gr.Blocks(title="aimusic") as demo:
        gr.Markdown(
            "# aimusic\nDescribe music in plain English and generate it locally. "
            "Early-stage: quality is okay, not high. A 5-minute track can take about 16 minutes on a 4 GB GPU."
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Describe the music", lines=2,
                    placeholder="e.g. 3 min soothing piano with rain, good for studying",
                )
                gr.Examples(examples=[[p] for p in EXAMPLE_PROMPTS], inputs=[prompt], label="Try one")
                backend = gr.Dropdown(BACKENDS, value=default_backend(), label="Backend", info=BACKEND_HELP)
                with gr.Accordion("Advanced settings", open=False):
                    duration = gr.Number(
                        value=0, precision=0, minimum=0, maximum=MAX_DURATION_S,
                        label="Length in seconds (0 = take it from the prompt, max 600)",
                    )
                    seed = gr.Number(value=None, precision=0, label="Seed (empty = random; same seed + prompt repeats a result)")
                    tuning = gr.Number(value=440, minimum=400, maximum=480, label="Tuning: concert A in Hz")
                    ambience_on = gr.Checkbox(value=True, label="Mix in rain/wind if the prompt asks for it")
                    level = gr.Slider(0, 1, value=0.35, step=0.05, label="Ambience level")
                    lyrics = gr.Textbox(
                        label="Lyrics (optional; acestep only, untested; leave empty for instrumental)",
                        lines=4, placeholder="[verse]\n...",
                    )
                with gr.Row():
                    go = gr.Button("Generate", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column(scale=2):
                preview = gr.Markdown(_preview_md("", 0, 440, True, 0.35))
        status = gr.Markdown()
        audio = gr.Audio(label="Result", type="filepath", interactive=False)
        with gr.Row():
            file_out = gr.File(label="Download")
            settings_out = gr.JSON(label="Settings used (saved next to the audio as .json)")

        preview_inputs = [prompt, duration, tuning, ambience_on, level]
        for comp in preview_inputs:
            comp.change(_preview_md, preview_inputs, preview, show_progress="hidden")

        run = go.click(
            generate,
            [prompt, backend, duration, seed, tuning, ambience_on, level, lyrics],
            [status, audio, settings_out, file_out],
            api_name="generate",
            concurrency_limit=1,
        )
        stop.click(lambda: "Stopping...", None, status, cancels=[run])
        demo.queue(default_concurrency_limit=1)
    return demo


def launch(port: int = 7860, host: str = "127.0.0.1", **kwargs) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_app().launch(
        server_name=host, server_port=port, inbrowser=True,
        allowed_paths=[str(OUTPUT_DIR.resolve())], **kwargs,
    )
