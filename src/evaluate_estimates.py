"""Offline estimate evaluator (no Webots required).

Scores our submitted victim_location_estimates.csv exactly the way the marking
supervisor does, so every pipeline change can be judged in seconds instead of a
full simulator run.

What it does:
  1. Parse every `Victim` node from a world .wbt (name, model, translation, yaw).
  2. Parse the `marker_offsets` table straight out of protos/Victim.proto (NOT
     hard-coded, so it tracks upstream changes).
  3. Compute each victim's TRUE marker world position:
        marker_world = node_translation + R(axis, angle) @ marker_offset
     (full axis-angle rotation via Rodrigues, so non-z tilt axes are handled).
  4. Load victim_location_estimates.csv, reproduce the marking server's transform
        world_estimate = csv_xy + origin_offset            (translation only!)
     match each estimate to its nearest true marker, and print the error table.
  5. FRAME AUDIT: also test the 90-degree-rotated interpretation of the CSV and
     report which one fits, so a frame regression is caught immediately.

Usage:
    python evaluate_estimates.py [world.wbt] [estimates.csv]
Defaults: worlds/large_world.wbt and the controller's sim_logs CSV.
"""
import os
import re
import sys
import math
import csv

from sar_pipeline import _COMP_ROOT, _SOLUTION_DIR   # shared path resolution
_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _COMP_ROOT

# Per-world layout: `python evaluate_estimates.py small_world` scores that world's
# archived estimates. With no argument it scores whichever world is CURRENTLY
# mirrored into sim_logs/ (i.e. the one the simulation would run).
DEFAULT_WORLD_NAME = "large_world"
_SIM_LOGS = os.path.join(_SOLUTION_DIR, "sim_logs")


def world_paths(name=None):
    """(world .wbt, estimates csv) for a world name, or the active sim_logs copy."""
    if name:
        return (os.path.join(_REPO_ROOT, "worlds", name + ".wbt"),
                os.path.join(_SIM_LOGS, name, "victim_location_estimates.csv"))
    return (os.path.join(_REPO_ROOT, "worlds", DEFAULT_WORLD_NAME + ".wbt"),
            os.path.join(_SIM_LOGS, "victim_location_estimates.csv"))


DEFAULT_WORLD = os.path.join(_REPO_ROOT, "worlds", DEFAULT_WORLD_NAME + ".wbt")
DEFAULT_CSV   = os.path.join(_SIM_LOGS, "victim_location_estimates.csv")
VICTIM_PROTO  = os.path.join(_REPO_ROOT, "protos", "Victim.proto")
SCORE_RADIUS_M = 1.0


# ----------------------------- parsing --------------------------------------
def parse_marker_offsets(proto_path):
    """Read the marker_offsets JS dict out of Victim.proto -> {model: (x,y,z)}."""
    with open(proto_path) as f:
        text = f.read()
    # grab the block between 'marker_offsets = {' and the closing '}'
    m = re.search(r"marker_offsets\s*=\s*\{(.*?)\}", text, re.S)
    block = m.group(1) if m else ""
    offsets = {}
    for model, vec in re.findall(r'"(\w+)"\s*:\s*"([-\d.\s]+)"', block):
        parts = [float(p) for p in vec.split()]
        while len(parts) < 3:
            parts.append(0.0)
        offsets[model] = tuple(parts[:3])
    return offsets


def parse_world_victims(world_path):
    """Return list of dicts: name, model, translation (x,y,z), axis (x,y,z), angle."""
    with open(world_path) as f:
        text = f.read()
    victims = []
    for block in re.findall(r"Victim\s*\{(.*?)\}", text, re.S):
        def field(pat, default=None):
            mm = re.search(pat, block)
            return mm.group(1) if mm else default
        name = field(r'name\s+"([^"]+)"')
        model = field(r'model\s+"([^"]+)"')
        tr = field(r"translation\s+([-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+)")
        rot = field(r"rotation\s+([-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+)")
        if tr is None:
            continue
        t = [float(v) for v in tr.split()]
        if rot is not None:
            r = [float(v) for v in rot.split()]
            axis, angle = r[:3], r[3]
        else:
            axis, angle = [0.0, 0.0, 1.0], 1.5708   # Victim.proto default rotation
        victims.append(dict(name=name, model=model, translation=t,
                            axis=axis, angle=angle))
    return victims


def parse_origin_offset(world_path):
    """OriginMarker world translation (the marking supervisor's coordinate_offset)."""
    with open(world_path) as f:
        text = f.read()
    m = re.search(r"OriginMarker\s*\{\s*translation\s+([-\d.eE]+)\s+([-\d.eE]+)", text)
    if not m:
        return (0.0, 0.0)
    return (float(m.group(1)), float(m.group(2)))


# ----------------------------- geometry -------------------------------------
def rodrigues(vec, axis, angle):
    """Rotate vec by angle (rad) about axis using Rodrigues' formula."""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-12:
        return tuple(vec)
    ax, ay, az = ax / n, ay / n, az / n
    c, s = math.cos(angle), math.sin(angle)
    vx, vy, vz = vec
    dot = ax * vx + ay * vy + az * vz
    # v*cos + (k x v)*sin + k*(k.v)*(1-c)
    cross = (ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx)
    rx = vx * c + cross[0] * s + ax * dot * (1 - c)
    ry = vy * c + cross[1] * s + ay * dot * (1 - c)
    rz = vz * c + cross[2] * s + az * dot * (1 - c)
    return (rx, ry, rz)


def true_marker(victim, offsets):
    off = offsets.get(victim["model"], (-1.0, 0.0, 0.0))   # proto's own fallback
    roff = rodrigues(off, victim["axis"], victim["angle"])
    t = victim["translation"]
    return (t[0] + roff[0], t[1] + roff[1])


def load_csv(csv_path):
    pts = []
    if not os.path.exists(csv_path):
        return pts
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                pts.append((float(row[0]), float(row[1])))
            except ValueError:
                continue
    return pts


# ----------------------------- evaluation -----------------------------------
def match_and_error(estimates_world, markers):
    """Greedy nearest-marker match; return list of (victim, est, err)."""
    used = set()
    out = []
    for est in estimates_world:
        best_i, best_d = None, float("inf")
        for i, (name, mk) in enumerate(markers):
            if i in used:
                continue
            d = math.hypot(est[0] - mk[0], est[1] - mk[1])
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
            out.append((markers[best_i][0], markers[best_i][1], est, best_d))
    return out


def mean_err(estimates_world, markers):
    matches = match_and_error(estimates_world, markers)
    if not matches:
        return float("inf")
    return sum(m[3] for m in matches) / len(matches)


def main():
    # Accept a bare world NAME ("small_world"), or explicit .wbt + csv paths.
    if len(sys.argv) > 1 and not sys.argv[1].endswith(".wbt"):
        world, csv_path = world_paths(sys.argv[1])
    else:
        world = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORLD
        csv_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CSV

    offsets = parse_marker_offsets(VICTIM_PROTO)
    victims = parse_world_victims(world)
    origin = parse_origin_offset(world)
    markers = [(v["name"], true_marker(v, offsets)) for v in victims]

    print(f"World : {os.path.relpath(world, _REPO_ROOT)}")
    print(f"CSV   : {os.path.relpath(csv_path, _REPO_ROOT)}")
    print(f"Origin offset (coordinate_offset): ({origin[0]:.3f}, {origin[1]:.3f})")

    # Self-test: the brief pins man_2's marker at ~(-5.01, -10.26).
    for v in victims:
        if v["model"] == "man_2":
            mk = true_marker(v, offsets)
            ok = abs(mk[0] + 5.01) < 0.03 and abs(mk[1] + 10.26) < 0.03
            print(f"[self-test] man_2 marker = ({mk[0]:.2f}, {mk[1]:.2f})  "
                  f"expected (-5.01, -10.26)  {'OK' if ok else 'MISMATCH -> rotation wrong!'}")

    print("\nTrue victim markers (world frame):")
    for name, mk in markers:
        print(f"  {name:8s} marker=({mk[0]:+.2f}, {mk[1]:+.2f})")

    csv_pts = load_csv(csv_path)
    print(f"\nEstimates submitted: {len(csv_pts)}   |   true victims: {len(markers)}")
    if len(csv_pts) != len(markers):
        print(f"  !! COUNT MISMATCH: penalised by Video Info Extraction scoring")

    # Marking server transform: world = csv + origin (translation only).
    as_is = [(x + origin[0], y + origin[1]) for (x, y) in csv_pts]
    # 90-deg rotated interpretation, for the frame audit.
    rot90 = [(y + origin[0], -x + origin[1]) for (x, y) in csv_pts]

    e_as_is = mean_err(as_is, markers)
    e_rot90 = mean_err(rot90, markers)
    print("\n=== FRAME AUDIT ===")
    print(f"  CSV read as origin-marker frame (correct): mean err {e_as_is:.2f} m")
    print(f"  CSV read as 90-deg-rotated                : mean err {e_rot90:.2f} m")
    if e_as_is <= e_rot90:
        print("  -> CSV appears to be in the ORIGIN-MARKER frame (good).")
        chosen = as_is
    else:
        print("  -> CSV appears ROTATED 90 deg: the frame fix has NOT been applied\n"
              "     to this CSV. Regenerate the pipeline or migrate the file.")
        chosen = as_is   # still report the as-the-server-would-see-it numbers

    print("\n=== PER-VICTIM (as the marking server scores it) ===")
    matches = match_and_error(chosen, markers)
    print(f"  {'victim':8s} {'true marker':>18s} {'our estimate':>18s} {'err(m)':>7s}  score")
    scored = 0
    for name, mk, est, err in matches:
        hit = err < SCORE_RADIUS_M
        scored += hit
        print(f"  {name:8s} ({mk[0]:+6.2f},{mk[1]:+6.2f})  ({est[0]:+6.2f},{est[1]:+6.2f})  "
              f"{err:6.2f}  {'WITHIN 1m' if hit else 'miss'}")
    print(f"\n  estimates within 1.0 m of a true marker: {scored}/{len(markers)}")
    print(f"  mean localization error: {e_as_is:.2f} m")


if __name__ == "__main__":
    main()
