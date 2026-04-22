"""Offline per-run analysis builder.

Imports the viewer app pointed at /tmp/data_all and dumps
_compute_run_analysis output for each GRPO run into
/tmp/data_all/run_analysis/<run_id>.json. Also writes
/tmp/data_all/run_analysis/_index.json listing available runs.

Mirrors build_flip_index.py - runs offline, ships result in the data
bundle, so the live Render instance never does the heavy compute.
"""
import os
import sys
import json
import time

os.environ["GRPO_RUN_DIRS"] = "/tmp/data_all/grpo"
os.environ["SFT_COMPARE_DIRS"] = "/tmp/data_all/sft"
sys.path.insert(
    0,
    "/sessions/laughing-eager-allen/mnt/grpo-tdc-gptoss-dequant-unsloth-16t-v15-v15-ep1-sr2-tis-colo-0420_1223/viewer",
)
os.chdir(
    "/sessions/laughing-eager-allen/mnt/grpo-tdc-gptoss-dequant-unsloth-16t-v15-v15-ep1-sr2-tis-colo-0420_1223/viewer"
)

t0 = time.time()
import app as _app  # noqa: E402
t1 = time.time()
print(f"[run_analysis] imported app in {t1 - t0:.1f}s; runs={_app.REG.run_order}")

out_dir = "/tmp/data_all/run_analysis"
os.makedirs(out_dir, exist_ok=True)

index_entries = []
total_samples = 0
for rid in _app.REG.run_order:
    ti = _app.REG.runs.get(rid)
    if ti is None:
        continue
    t2 = time.time()
    res = _app._compute_run_analysis(ti)
    t3 = time.time()

    res_slim = {k: v for k, v in res.items() if k != "samples"}
    res_slim["n_total"] = res.get("n_total", len(res.get("samples", [])))

    out_path = os.path.join(out_dir, f"{rid}.json")
    with open(out_path, "w") as f:
        json.dump(res_slim, f)

    # Unload samples after each run to keep peak RAM low during the build.
    for g in ti.groups:
        try:
            g.unload_samples()
        except AttributeError:
            pass

    size = os.path.getsize(out_path)
    n_steps = len(res_slim.get("steps", []))
    n_total = res_slim["n_total"]
    total_samples += n_total
    print(
        f"[run_analysis] {rid}: {n_steps} steps, {n_total} samples, "
        f"computed in {t3 - t2:.1f}s, wrote {size/1024:.1f} KB"
    )
    index_entries.append({
        "run_id": rid,
        "n_total": n_total,
        "n_steps": n_steps,
        "file": f"{rid}.json",
    })

idx_path = os.path.join(out_dir, "_index.json")
with open(idx_path, "w") as f:
    json.dump({"runs": index_entries}, f, indent=2)
print(
    f"[run_analysis] done: {len(index_entries)} runs, {total_samples} samples, "
    f"index at {idx_path}"
)
