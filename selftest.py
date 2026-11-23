import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

import facial_analysis as fa
import metrics
import version
from feature_extractor import FeatureExtractor

METRICS = ["rest_idx", "droop", "brow_rest_asym", "smile_idx", "smile_asym", "brow_raise_asym",
           "eye_close_asym", "lower_idx", "upper_idx", "smile_onset_lag_s", "ptosis_asym",
           "blink_rate_per_min", "blink_lag_ms", "blink_depth_asym"]
RUN_COLUMNS = ["time", "subject", "condition", "rep", "n_frames", "valid_frac", "pattern",
               "nihss4", "ptosis_flag", "fingerprint"] + METRICS
FRAME_COLUMNS = ["subject", "condition", "rep", "stage", "yaw", "pitch", "roll", "ocw_px",
                 "droop", "brow_rest_asym", "earL", "earR"]

THRESH = {"lower_idx": fa.LOWER_MILD, "upper_idx": fa.UPPER_FLAG}


def run_rep(frames, subject, condition, rep, max_s=45.0):
    ex = FeatureExtractor()
    t_start = None
    rows = []
    n = valid = 0
    last = None
    for t, lms, w, h in frames:
        if t_start is None:
            t_start = t
            ex.facial.start_exam(t=t)
        f = ex.extract(lms, w, h, t=t)
        fac = f["facial"]
        n += 1
        valid += int(fac["valid"])
        last = fac
        if fac["valid"]:
            rows.append({"subject": subject, "condition": condition, "rep": rep,
                         "stage": fac["stage"], "yaw": round(f["yaw"], 2),
                         "pitch": round(f["pitch"], 2), "roll": round(f["roll"], 2),
                         "ocw_px": round(float(np.hypot(
                             (lms[fa.OUTER_R].x - lms[fa.OUTER_L].x) * w,
                             (lms[fa.OUTER_R].y - lms[fa.OUTER_L].y) * h)), 1),
                         "droop": round(fac["droop"], 4),
                         "brow_rest_asym": round(fac["brow_rest_asym"], 4),
                         "earL": round(f["ear_left"], 4), "earR": round(f["ear_right"], 4)})
        if fac["stage"] == "DONE" or t - t_start > max_s:
            break
    if last is None:
        return None, []
    run = {"time": datetime.now().isoformat(timespec="seconds"), "subject": subject,
           "condition": condition, "rep": rep, "n_frames": n,
           "valid_frac": round(valid / max(n, 1), 3), "pattern": last["pattern"],
           "nihss4": last["nihss4"], "ptosis_flag": int(bool(last.get("ptosis_flag"))),
           "fingerprint": version.source_fingerprint()}
    for k in METRICS:
        v = last.get(k)
        run[k] = "" if v is None else round(float(v), 5)
    return run, rows


def _append(path, columns, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        if new:
            w.writeheader()
        w.writerows(rows)


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, ValueError, TypeError):
            pass
    return out


def report(out_dir):
    runs = _read(os.path.join(out_dir, "selftest_runs.csv"))
    frames = _read(os.path.join(out_dir, "selftest_frames.csv"))
    if not runs:
        return "No self-test runs found."
    lines = []
    subs = sorted({r["subject"] for r in runs})
    conds = sorted({r["condition"] for r in runs})
    fps = sorted({r["fingerprint"] for r in runs})
    lines.append(f"# Face self-test report\n\n{len(runs)} exams, {len(subs)} subjects, "
                 f"conditions: {', '.join(conds)}\n")
    lines.append("All subjects are assumed HEALTHY. Anything flagged below is a false positive.\n")
    if len(fps) > 1:
        lines.append(f"**Warning: runs come from {len(fps)} different code versions.**\n")

    lines.append("## Spread of each metric on healthy faces\n")
    lines.append("| Metric | n | mean | SD | p95 | max | threshold | headroom |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in METRICS:
        v = _num(runs, k)
        if not v:
            continue
        p95 = float(np.percentile(v, 95))
        thr = THRESH.get(k)
        head = ""
        if thr:
            head = f"{(thr - p95) / thr:+.0%}"
        lines.append(f"| {k} | {len(v)} | {np.mean(v):.4f} | {np.std(v):.4f} | {p95:.4f} | "
                     f"{max(v):.4f} | {thr if thr else ''} | {head} |")
    lines.append("\nHeadroom = how far the 95th percentile sits below the threshold. "
                 "Under about +30% means healthy faces already come close to triggering it.\n")

    lines.append("## False-positive rate on healthy exams\n")
    lines.append("| Rule | flagged | rate | 95% CI |")
    lines.append("|---|---|---|---|")
    n = len(runs)
    rules = {
        f"lower_idx >= {fa.LOWER_MILD}": sum(1 for r in runs if _f(r, "lower_idx") >= fa.LOWER_MILD),
        f"upper_idx >= {fa.UPPER_FLAG}": sum(1 for r in runs if _f(r, "upper_idx") >= fa.UPPER_FLAG),
        "pattern CENTRAL": sum(1 for r in runs if r["pattern"] == "CENTRAL"),
        "pattern not NORMAL/INSUFFICIENT": sum(1 for r in runs if r["pattern"] not in ("NORMAL", "INSUFFICIENT")),
        "ptosis flagged": sum(int(r["ptosis_flag"] or 0) for r in runs),
    }
    for name, k in rules.items():
        lo, hi = metrics.wilson(k, n)
        lines.append(f"| {name} | {k}/{n} | {k / n:.0%} | {lo:.0%}-{hi:.0%} |")
    lines.append("")

    lines.append("## By condition (mean lower_idx / upper_idx / valid fraction)\n")
    lines.append("| Condition | n | lower_idx | upper_idx | ptosis_asym | valid frames |")
    lines.append("|---|---|---|---|---|---|")
    for c in conds:
        rs = [r for r in runs if r["condition"] == c]
        lines.append(f"| {c} | {len(rs)} | {_m(rs, 'lower_idx')} | {_m(rs, 'upper_idx')} | "
                     f"{_m(rs, 'ptosis_asym')} | {_m(rs, 'valid_frac')} |")
    lines.append("")

    lines.append("## Test-retest (within one subject and condition)\n")
    sds = {k: [] for k in ("lower_idx", "upper_idx", "droop", "smile_asym", "brow_raise_asym")}
    for s in subs:
        for c in conds:
            rs = [r for r in runs if r["subject"] == s and r["condition"] == c]
            if len(rs) < 2:
                continue
            for k in sds:
                v = _num(rs, k)
                if len(v) >= 2:
                    sds[k].append(float(np.std(v, ddof=1)))
    if any(sds.values()):
        lines.append("| Metric | pooled within-subject SD | groups |")
        lines.append("|---|---|---|")
        for k, v in sds.items():
            if v:
                lines.append(f"| {k} | {np.sqrt(np.mean(np.square(v))):.4f} | {len(v)} |")
    else:
        lines.append("Need at least 2 repetitions per subject and condition.")
    lines.append("")

    lines.append("## Resting droop vs head yaw\n")
    rest = [r for r in frames if r["stage"] in ("REST", "IDLE")]
    if rest:
        lines.append("| yaw bin (deg) | frames | mean droop | SD |")
        lines.append("|---|---|---|---|")
        for lo in range(-20, 20, 5):
            sel = [float(r["droop"]) for r in rest if lo <= float(r["yaw"]) < lo + 5]
            if len(sel) >= 5:
                lines.append(f"| {lo:+d} to {lo + 5:+d} | {len(sel)} | {np.mean(sel):+.4f} | {np.std(sel):.4f} |")
        lines.append("\nIf the mean drifts across bins, head turn is leaking into the droop measure. "
                     f"The droop normal limit is {fa.REST_DROOP_NORMAL}.")
    else:
        lines.append("No frame data.")
    return "\n".join(lines)


def _f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return 0.0


def _m(rs, k):
    v = _num(rs, k)
    return f"{np.mean(v):.3f}" if v else "-"


def live_frames(spec, prompt_fn=None, max_s=45.0):
    import cv2
    import mediapipe as mp
    from capture import FrameSource
    src = FrameSource(spec)
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def gen():
        t0 = None
        try:
            while True:
                ok, fr = src.read()
                if not ok or fr is None:
                    return
                h, w = fr.shape[:2]
                res = mesh.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                t = time.monotonic()
                if res.multi_face_landmarks:
                    t0 = t0 or t
                    if prompt_fn:
                        if prompt_fn(fr, t - t0) is False:
                            return
                    yield t, res.multi_face_landmarks[0].landmark, w, h
                else:
                    if prompt_fn and prompt_fn(fr, None) is False:
                        return
        finally:
            src.release()
    return gen()


def cmd_run(a):
    import cv2
    os.makedirs(a.out, exist_ok=True)
    from capture import parse_source
    spec = parse_source(a.source, a.rotate, a.mirror)
    stop = {"q": False}
    state = {"stage": "IDLE"}

    def show(fr, t):
        txt = fa.PROMPT.get(state["stage"], "")
        cv2.putText(fr, f"{a.subject} / {a.condition}  {txt}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if t is None:
            cv2.putText(fr, "no face", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("self-test", fr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop["q"] = True
            return False
        return True

    for rep in range(1, a.reps + 1):
        if stop["q"]:
            break
        print(f"Rep {rep}/{a.reps}: get ready, starting in {a.delay}s")
        time.sleep(a.delay)
        frames = live_frames(spec, show)

        def tracked():
            for item in frames:
                yield item

        run, rows = run_rep(_stage_tap(tracked(), state), a.subject, a.condition, rep)
        if run is None:
            print("no usable frames")
            continue
        _append(os.path.join(a.out, "selftest_runs.csv"), RUN_COLUMNS, [run])
        _append(os.path.join(a.out, "selftest_frames.csv"), FRAME_COLUMNS, rows)
        frames.close()
        print(f"  pattern={run['pattern']} lower={run['lower_idx']} upper={run['upper_idx']} "
              f"valid={run['valid_frac']}")
    cv2.destroyAllWindows()


def _stage_tap(frames, state):
    ex_stage = ["REST", "SMILE", "BROWS", "EYES", "DONE"]
    t0 = None
    for item in frames:
        t = item[0]
        t0 = t0 or t
        el = t - t0
        acc = 0.0
        st = "DONE"
        for s in ex_stage[:-1]:
            acc += fa.DURATION[s]
            if el < acc:
                st = s
                break
        state["stage"] = st
        yield item


def main(argv=None):
    ap = argparse.ArgumentParser(description="Healthy-face noise-floor and false-positive self-test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--subject", required=True)
    r.add_argument("--condition", default="normal")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--source", default="0")
    r.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270))
    r.add_argument("--mirror", action="store_true")
    r.add_argument("--delay", type=float, default=3.0)
    r.add_argument("--out", default="selftest_data")
    p = sub.add_parser("report")
    p.add_argument("--out", default="selftest_data")
    p.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    else:
        text = report(a.out)
        print(text)
        if a.save:
            with open(a.save, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
