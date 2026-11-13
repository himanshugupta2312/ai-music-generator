"""Local web UI (Gradio). Start with:  aimusic ui

Runs only on this machine (127.0.0.1). One generation at a time, because the GPU
is the bottleneck and a 4 GB card cannot hold two models.
"""

from __future__ import annotations

import gc
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from .backends import get_backend
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

# (button label, prompt). Short lengths on purpose so a first try is quick.
VIBES = [
    ("📚 Study time", "1 min calm soothing piano music with rain for studying"),
    ("🏋️ Workout", "45 sec punk rock with distorted guitars and fast drums"),
    ("☕ Lofi café", "45 sec lofi hip hop beats, mellow, 85 bpm"),
    ("🎬 Movie trailer", "45 sec epic cinematic music"),
    ("🌙 Sleepy time", "2 min sleep music with soft piano and wind"),
    ("🎷 Jazz night", "45 sec mellow jazz with soft piano, 70 bpm"),
    ("🌲 Forest floor", "1 min soothing music, theme: sitting on a forest floor, with rain"),
    ("🎛️ Genre mash-up", "45 sec soundtrack with hip hop beats and punk music at 450hz"),
]

_LENGTHS = ["30 sec", "45 sec", "1 min", "90 sec"]
_STYLES = [
    "calm piano", "lofi hip hop beats", "punk rock", "epic cinematic music", "mellow jazz",
    "dark trap beats", "happy upbeat tune", "soothing ambient music", "sad piano",
]
_EXTRAS = ["", "", " with rain", " with wind", " for studying", " for a rainy night"]
_TEMPOS = ["", "", ", 70 bpm", ", 90 bpm", ", 120 bpm"]

FUN_LINES = [
    "🎹 Warming up the piano…",
    "🥁 Convincing the drummer to keep time…",
    "🎸 Tuning the guitars…",
    "🎛️ Twiddling the knobs…",
    "🎻 Politely asking the violins to stop squeaking…",
    "☕ Brewing coffee for the band…",
    "✨ Sprinkling a little reverb…",
    "🎧 Still cooking, good music takes a moment…",
]

BACKEND_CHOICES = [
    ("🎹 ACE-Step: one long, continuous piece (needs the separate install)", "acestep"),
    ("🎛️ MusicGen: short 30 s clips (longer ones are stitched)", "musicgen"),
    ("⚡ Quick synth: instant, simple sound, no model needed", "synth"),
]

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="pink",
    neutral_hue="slate",
    radius_size="lg",
)

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#hero {
  text-align: center; padding: 26px 16px 22px; border-radius: 22px; color: #fff;
  background: linear-gradient(120deg, #7c3aed 0%, #db2777 55%, #f59e0b 100%);
  box-shadow: 0 10px 30px rgba(124, 58, 237, .28);
}
#hero h1 { margin: 0; font-size: 2.5rem; letter-spacing: -.02em; color: #fff; }
#hero p { margin: 6px 0 0; font-size: 1.08rem; opacity: .95; color: #fff; }
#hero .chips { margin-top: 12px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
#hero .chip { background: rgba(255,255,255,.2); border-radius: 999px; padding: 3px 12px; font-size: .85rem; color: #fff; }
.vibe button, button.vibe {
  min-height: 72px; font-size: 1.02rem; border-radius: 16px;
  transition: transform .14s ease, box-shadow .14s ease;
}
.vibe button:hover, button.vibe:hover { transform: translateY(-3px) rotate(-.8deg); box-shadow: 0 8px 18px rgba(124,58,237,.25); }
#go { min-height: 54px; font-size: 1.15rem; font-weight: 700; }
#go:not(:disabled) { animation: pulse 2.6s ease-in-out infinite; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(219,39,119,.45); } 50% { box-shadow: 0 0 0 10px rgba(219,39,119,0); } }
@media (prefers-reduced-motion: reduce) {
  #go:not(:disabled) { animation: none; }
  .vibe button:hover, button.vibe:hover { transform: none; }
}
.card { border-radius: 16px; padding: 4px 8px; }
#footnote { text-align: center; opacity: .75; font-size: .88rem; padding: 8px 0 18px; }
"""


def surprise_prompt(rng: random.Random | None = None) -> str:
    """A random, always-valid prompt for the Surprise me button."""
    r = rng or random
    return f"{r.choice(_LENGTHS)} {r.choice(_STYLES)}{r.choice(_EXTRAS)}{r.choice(_TEMPOS)}"


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _preview_md(prompt, duration, tuning, ambience_on, level) -> str:
    """Show how the prompt will be read (instant, no model, no LLM)."""
    if not prompt or not prompt.strip():
        return "### 🔍 How I'll read it\n*Pick a vibe or type something, and I'll show you what I understood.*"
    s = build_settings(
        prompt, duration=duration or None, tuning=tuning or None,
        ambience=bool(ambience_on), ambience_level=level,
    )
    rows = "\n".join(f"- **{label}:** {value}" for label, value in describe(s))
    note = ""
    if s.duration > MAX_DURATION_S:
        note = f"\n\n> ⚠️ Over the {MAX_DURATION_S // 60}-minute limit: shorten it before generating."
    return "### 🔍 How I'll read it\n" + rows + note


def _cleanup_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(prompt, backend_name, duration, seed, tuning, ambience_on, level, lyrics,
             history, progress=gr.Progress()):
    """Generator. Yields (status, audio, settings, file, history_state, history_dropdown)."""
    if not prompt or not prompt.strip():
        raise gr.Error("Pick a vibe or describe the music first.")
    history = list(history or [])
    try:
        s = build_settings(
            prompt,
            duration=duration or None,
            seed=random.randint(1, 99999) if seed in (None, "") else int(seed),
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
    first = True
    try:
        while thread.is_alive():
            elapsed = time.time() - start
            line = FUN_LINES[int(elapsed // 6) % len(FUN_LINES)]
            progress(min(elapsed / est, 0.97),
                     desc=f"{line}  {_fmt_time(elapsed)} elapsed, rough estimate {_fmt_time(est)}")
            yield (f"**{line}**  \n{_fmt_time(elapsed)} elapsed with **{backend_name}**. "
                   f"The estimate ({_fmt_time(est)}) is rough and depends on your hardware."
                   + warn_text,
                   None if first else gr.skip(), None if first else gr.skip(),
                   gr.DownloadButton(interactive=False) if first else gr.skip(), gr.skip(), gr.skip())
            first = False
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
    path = str(out.resolve())
    label = f"{datetime.now():%H:%M} · {prompt.strip()[:48]}"
    history.insert(0, [label, path])
    yield (f"### 🎉 Ta-da! Done in {_fmt_time(took)}\nSaved to `{out}`. "
           "Not quite right? Try **Another take** for a fresh version of the same idea." + warn_text,
           path, s.to_dict(), gr.DownloadButton(value=path, interactive=True), history,
           gr.Dropdown(choices=[(l, p) for l, p in history], value=path))


def remix(prompt, backend_name, duration, seed, tuning, ambience_on, level, lyrics,
          history, progress=gr.Progress()):
    """Same request with a fresh random seed."""
    yield from generate(prompt, backend_name, duration, None, tuning, ambience_on, level,
                        lyrics, history, progress)


def _setter(text: str):
    return lambda: text


def build_app() -> gr.Blocks:
    with gr.Blocks(title="aimusic") as demo:
        gr.HTML(
            '<div id="hero"><h1>🎵 aimusic</h1>'
            "<p>Tell me a vibe. I'll make a tune.</p>"
            '<div class="chips"><span class="chip">🔒 runs on your computer</span>'
            '<span class="chip">🎲 every take is different</span>'
            '<span class="chip">📦 open source</span></div></div>'
        )

        gr.Markdown("### 🎭 Pick a vibe")
        vibe_buttons = []
        for row_start in range(0, len(VIBES), 4):
            with gr.Row():
                for label, _prompt in VIBES[row_start:row_start + 4]:
                    vibe_buttons.append(gr.Button(label, elem_classes=["vibe"]))

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    prompt = gr.Textbox(
                        label="…or describe your own", lines=2, scale=5,
                        placeholder="e.g. 3 min soothing piano with rain, good for studying",
                    )
                    surprise = gr.Button("🎲 Surprise me!", scale=2, min_width=170)
                backend = gr.Dropdown(BACKEND_CHOICES, value=default_backend(), label="Who's playing?")
                with gr.Accordion("🎚️ Advanced settings", open=False):
                    duration = gr.Number(
                        value=0, precision=0, minimum=0, maximum=MAX_DURATION_S,
                        label="Length in seconds (0 = take it from the prompt, max 600)",
                    )
                    with gr.Row():
                        seed = gr.Number(value=None, precision=0, scale=4,
                                         label="Seed (empty = surprise me; same seed + prompt repeats a result)")
                        dice = gr.Button("🎲", scale=1, min_width=60)
                    tuning = gr.Number(value=440, minimum=400, maximum=480, label="Tuning: concert A in Hz")
                    ambience_on = gr.Checkbox(value=True, label="🌧️ Mix in rain/wind if the prompt asks for it")
                    level = gr.Slider(0, 1, value=0.35, step=0.05, label="Ambience level")
                    lyrics = gr.Textbox(
                        label="Lyrics (optional; ACE-Step only, untested; empty = instrumental)",
                        lines=4, placeholder="[verse]\n...",
                    )
                with gr.Row():
                    go = gr.Button("✨ Make music!", variant="primary", elem_id="go", scale=3)
                    again = gr.Button("🔁 Another take", scale=2)
                    stop = gr.Button("⏹ Stop", variant="stop", scale=1, min_width=90)
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["card"]):
                    preview = gr.Markdown(_preview_md("", 0, 440, True, 0.35))

        status = gr.Markdown()
        audio = gr.Audio(
            label="🎧 Your track", type="filepath", interactive=False,
            waveform_options=gr.WaveformOptions(waveform_color="#a78bfa", waveform_progress_color="#db2777"),
        )
        with gr.Row():
            file_out = gr.DownloadButton("⬇️ Download your track", interactive=False)
            history_dd = gr.Dropdown(choices=[], label="🕘 Your tracks this session (pick one to replay)")
        with gr.Accordion("🧾 The recipe (settings used)", open=False):
            settings_out = gr.JSON(label="Saved next to the audio as .json, so you can reproduce it")
        history = gr.State([])
        gr.HTML('<div id="footnote">Early-stage project: results are okay, not studio quality. '
                "A long track can take a while on a small GPU, so the timer is only a rough guide.</div>")

        # vibes and surprise fill the prompt box
        for btn, (_label, text) in zip(vibe_buttons, VIBES):
            btn.click(_setter(text), None, prompt, show_progress="hidden")
        surprise.click(lambda: surprise_prompt(), None, prompt, show_progress="hidden")
        dice.click(lambda: random.randint(1, 99999), None, seed, show_progress="hidden")

        preview_inputs = [prompt, duration, tuning, ambience_on, level]
        for comp in preview_inputs:
            comp.change(_preview_md, preview_inputs, preview, show_progress="hidden")

        run_inputs = [prompt, backend, duration, seed, tuning, ambience_on, level, lyrics, history]
        run_outputs = [status, audio, settings_out, file_out, history, history_dd]
        run = go.click(generate, run_inputs, run_outputs, api_name="generate",
                       concurrency_limit=1, concurrency_id="gpu")
        run2 = again.click(remix, run_inputs, run_outputs, api_name="remix",
                           concurrency_limit=1, concurrency_id="gpu")
        stop.click(lambda: "⏹ Stopping…", None, status, cancels=[run, run2])
        history_dd.input(lambda p: p, history_dd, audio, show_progress="hidden")
        demo.queue(default_concurrency_limit=1)
    return demo


def launch(port: int = 7860, host: str = "127.0.0.1", inbrowser: bool = True, **kwargs) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_app().launch(
        server_name=host, server_port=port, inbrowser=inbrowser,
        allowed_paths=[str(OUTPUT_DIR.resolve())], theme=THEME, css=CSS, **kwargs,
    )
