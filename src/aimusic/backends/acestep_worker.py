"""Runs INSIDE the ACE-Step virtual environment (not aimusic's).

Launched by AceStepBackend as:  <acestep-python> -I acestep_worker.py <job.json>
Prints a line "AUDIO <path>" on success. Kept free of aimusic imports on purpose,
because ACE-Step pins its own torch version and lives in a separate environment.
"""

import json
import os
import sys
import time


def main() -> int:
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    ace_home = job["ace_home"]
    sys.path.insert(0, ace_home)

    import torch

    # Pre-Ampere GPUs (e.g. Quadro T2000, GTX 16xx/RTX 20xx) get float16 in ACE-Step,
    # which overflows to NaN latents. bfloat16 has fp32 range and avoids it (verified on a
    # Quadro T2000: float16 -> NaN, bfloat16 -> valid audio).
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        import acestep.core.generation.handler.memory_utils as _m
        import acestep.gpu_config as _g

        _g.cuda_supports_bfloat16 = lambda *a, **k: True
        _m.cuda_supports_bfloat16 = lambda *a, **k: True
        print("[worker] pre-Ampere GPU: forcing bfloat16", flush=True)

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    handler = AceStepHandler()
    t0 = time.time()
    ok = False
    # Low-VRAM settings (tier <= 6 GB per ACE-Step's GPU guide): INT8 + CPU offload, no LM.
    for quant in ("int8_weight_only", None):
        msg, ok = handler.initialize_service(
            project_root=ace_home,
            config_path="acestep-v15-turbo",
            device="auto",
            offload_to_cpu=True,
            offload_dit_to_cpu=True,
            quantization=quant,
        )
        print(f"[worker] init quant={quant} ok={ok} ({time.time() - t0:.0f}s)", flush=True)
        if ok:
            break
    if not ok:
        print("ERROR model init failed", flush=True)
        return 2

    seed = job.get("seed")
    params = GenerationParams(
        task_type="text2music",
        thinking=False,
        caption=job["caption"],
        lyrics=job["lyrics"],
        instrumental=job["instrumental"],
        duration=float(job["duration"]),
        inference_steps=8,
        guidance_scale=1.0,
        seed=-1 if seed is None else int(seed),
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format="wav",
        use_random_seed=seed is None,
        seeds=None if seed is None else [int(seed)],
    )
    t1 = time.time()
    res = generate_music(handler, None, params, config, save_dir=job["out_dir"])
    print(f"[worker] generated in {time.time() - t1:.0f}s", flush=True)
    if not res.success or not res.audios:
        print("ERROR", getattr(res, "error", None) or getattr(res, "status_message", None), flush=True)
        return 3
    print("AUDIO", res.audios[0].get("path"), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("ACESTEP_GENERATION_TIMEOUT", "3600")
    raise SystemExit(main())
