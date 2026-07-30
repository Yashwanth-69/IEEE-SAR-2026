#!/usr/bin/env python3
"""
sar_pipeline.py — the whole thing in one file.

    video + IMU csv
        -> STAGE 1  visual odometry          (final_10_consistent)
        -> STAGE 2  wall grid from frames    (final_segmentation_v23, verbatim)
        -> STAGE 3  ROSE intensity filter + rose2 vectorize + doors + render
        -> STAGE 4  score against ground truth (optional, needs walls.csv)

Run it with no arguments:

    python sar_pipeline.py

Every input path and every tuned parameter lives in the CONFIG block below.
Stages already computed are skipped automatically (delete their outputs, or
flip the FORCE_* flags, to redo them): stage 1 is skipped if
camera_poses.csv exists, stage 2 if wall_raw.npy exists. Stage 3 always
runs (seconds). Stage 4 runs if GT_CSV exists.

Stage parameters are the frozen values from the tune_vs_gt session.
"""

import csv
import json
import math
import os

import cv2
import numpy as np

# =============================== CONFIG ======================================
# Every path is anchored to this file's own location, so the pipeline behaves
# the same no matter which directory you launch it from. _SRC_DIR is the src/
# folder that holds this script and the trained models; _COMP_ROOT is the
# competition repo folder (its parent).
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
# This package lives inside the controller folder, so the controller folder (which
# owns sim_logs/) is one level up.
_SOLUTION_DIR = os.path.dirname(_SRC_DIR)


def _find_competition_root(start):
    """Walk up from `start` until we find the Webots competition folder, i.e. the
    one holding `recordings/` and `worlds/`. Resolving it by content rather than by
    a fixed number of parent directories keeps the paths correct wherever this
    package is checked out."""
    d = start
    for _ in range(6):
        if all(os.path.isdir(os.path.join(d, s)) for s in ("recordings", "worlds")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(_SOLUTION_DIR)     # sensible fallback


_COMP_ROOT = _find_competition_root(_SRC_DIR)

# ---- inputs ----
# CHANGE ONLY THIS LINE to switch worlds. The IMU csv, the per-world cache folder
# and the per-world output folder are all derived from it, so every world keeps its
# own results and an already-processed world replays instantly.
VIDEO_PATH  = os.path.join(_COMP_ROOT, "recordings", "large_world_flyover.mp4")

# "large_world_flyover.mp4" -> "large_world". Used as the per-world folder name.
WORLD = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
if WORLD.endswith("_flyover"):
    WORLD = WORLD[:-len("_flyover")]

IMU_CSV     = os.path.splitext(VIDEO_PATH)[0] + ".csv"        # same stem as the video
MODEL_PATH  = os.path.join(_SRC_DIR, "best_segmentation.pt")   # wall segmentation model
GT_CSV      = os.path.join(_COMP_ROOT, WORLD + "_walls.csv")   # ground truth (optional)

# ---- intermediates: per-world cache under src/<world>/ ----
# Keeping these per world is what lets you flip VIDEO_PATH between worlds without
# destroying the previous world's odometry/wall cache, so a world you have already
# processed replays instantly instead of re-running the long video pass.
WORLD_SRC_DIR = os.path.join(_SRC_DIR, WORLD)
MAPPING_DIR = os.path.join(WORLD_SRC_DIR, "mapping_data")
FRAMES_DIR  = os.path.join(MAPPING_DIR, "captured_frames")
CSV_PATH    = os.path.join(MAPPING_DIR, "camera_poses.csv")
WALL_NPY    = os.path.join(MAPPING_DIR, "wall_raw.npy")
TUNING_DIR  = os.path.join(MAPPING_DIR, "tuning")
COMPARE_DIR = os.path.join(MAPPING_DIR, "compare")

# ---- deliverables ----
# IMPORTANT: the marking supervisor reads these from FIXED paths that we must not
# change (sar_marking_supervisor.py hardcodes
# "../proposed_solution/sim_logs/victim_location_estimates.csv" and
# ".../map_estimate.png"). So the per-world folder is the ARCHIVE, and whichever
# world is active is mirrored up into sim_logs/ itself for the supervisor and the
# ground controller to read. activate_world() does that mirroring.
SIM_LOGS_DIR   = os.path.join(_SOLUTION_DIR, "sim_logs")
WORLD_LOGS_DIR = os.path.join(SIM_LOGS_DIR, WORLD)

OUT_PATH      = os.path.join(WORLD_LOGS_DIR, "map_estimate.png")
WALLS_OUT_CSV = os.path.join(WORLD_LOGS_DIR, "wall_estimates.csv")

# Files mirrored from the per-world archive into sim_logs/ for the active world.
DELIVERABLES = ("victim_location_estimates.csv", "wall_estimates.csv",
                "map_estimate.png", "map_estimate_meta.json",
                "victim_uncertainty.csv")


def activate_world(world=None):
    """Copy a world's archived deliverables up into sim_logs/ so the supervisor
    and the ground controller (which both read fixed paths) use that world.
    Returns True if the world had a usable set of deliverables."""
    import shutil
    world = world or WORLD
    src = os.path.join(SIM_LOGS_DIR, world)
    if not os.path.isdir(src):
        return False
    os.makedirs(SIM_LOGS_DIR, exist_ok=True)
    copied = 0
    for name in DELIVERABLES:
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(SIM_LOGS_DIR, name))
            copied += 1
    if copied:
        # Stamp which world the root copies belong to. The ground controller reads
        # this and prints it at startup, so opening the wrong .wbt is obvious
        # immediately instead of silently navigating to another world's victims.
        with open(os.path.join(SIM_LOGS_DIR, "ACTIVE_WORLD.txt"), "w") as f:
            f.write(world + "\n")
        print("[world] activated '%s' -> mirrored %d deliverable(s) into sim_logs/"
              % (world, copied))
    return copied > 0


def world_is_cached(world=None):
    """True when this world already has the deliverables the simulation needs, so
    the pipeline can be skipped entirely and you can go straight to Webots."""
    world = world or WORLD
    src = os.path.join(SIM_LOGS_DIR, world)
    need = ("victim_location_estimates.csv", "wall_estimates.csv", "map_estimate.png")
    return all(os.path.exists(os.path.join(src, n)) for n in need)

# ---- victim detection, fused into the single odometry video pass ----
# No frames are written to disk: each non-rotating frame is scored for victims
# using the same pose that odometry just computed for it, then all hits are
# clustered after the pass. This absorbs victims_location.py into the pipeline.
DETECT_VICTIMS     = True
VICTIM_MODEL_PATH  = os.path.join(_SRC_DIR, "yolo11m.pt")  # COCO person detector (class 0); swap to yolo11s.pt for speed
LAUNCH_EXCLUDE_M   = 1.5      # keep-out radius around origin: parked ROSbots on the launch pad are not victims
VICTIM_OUT_CSV     = os.path.join(WORLD_LOGS_DIR, "victim_location_estimates.csv")
# Per-victim uncertainty, written alongside the contract CSV (which must stay a
# bare "x,y" list). Row i describes victim i of that file: the 1-sigma major and
# minor axes of the localization uncertainty ellipse and the unit vector along
# the WEAK (major) axis. The ground controller can search along that direction
# instead of guessing a circle when a victim is not where the estimate says.
VICTIM_UNC_CSV     = os.path.join(WORLD_LOGS_DIR, "victim_uncertainty.csv")
VICTIM_CONF        = 0.60     # per-detection YOLO confidence floor: cull low/mid-confidence top-down false positives
VICTIM_CLASSES     = [0]      # class ids kept (COCO 'person' -> 0)
CLUSTER_RADIUS_M   = 1.5      # merge detections within this many metres
VICTIM_MERGE_RADIUS_M = 1.5   # second pass: merge whole clusters whose centres are within this
                              # (one large/lying victim can seed two clusters). Kept conservative
                              # so two genuinely distinct victims (metres apart here) are NOT merged.
MIN_DETECTION_HITS = 40       # a cluster needs this many frames to count (noise filter)
EXPECTED_VICTIM_COUNT = None  # set to the known victim count to keep only the top-N clusters
# Robust localization: averaging ground projections biases each victim by
# h*tan(beta) (body height x off-nadir angle). Instead we solve a Huber-robust
# least-squares closest point to all a cluster's viewing rays (multi-view
# triangulation), recovering the true 3D point so the z=0 parallax bias cancels.
HUBER_DELTA_M = 0.40   # ray residual (m) beyond which a detection is down-weighted
TRI_TRUST_M   = 1.50   # accept triangulation only if within this of the robust median
TRI_MAX_Z_M   = 3.00   # reject a triangulated height implausible for a body
IRLS_ITERS    = 6
# Degeneracy handling. When a victim is only ever seen over a NARROW viewing arc
# its relative viewing geometry barely changes, the target's HEIGHT stops being
# observable, and the position error grows without bound: the textbook
# bearing-only failure, and what leaves ~1.2 m residual on victim3/victim5. So we
# MEASURE it rather than silently falling back.
#
# The observable to test is the PARALLAX SPREAD. For each detection form
#     p_i = (ground projection - camera xy) / altitude = tan(beta) * azimuth
# and take the weighted covariance of p across the cluster. (The eigenvalues of
# the ray matrix A itself are NOT a usable test here: a near-nadir camera keeps
# A's xy block well conditioned no matter how poor the baseline is. Verified
# numerically.) If p never varies, every view carries the SAME h*tan(beta)
# offset, which no averaging and no triangulation can separate from the truth.
#
# Measured on synthetic flyovers: spread >= 0.02 gives triangulation error under
# 2 cm; below it the error climbs to 0.3-0.5 m while the raw projections sit
# more than a metre out. 0.02 of variance is a std of ~0.14 in tan(beta), about
# 8 degrees of viewing-angle variation.
PARALLAX_MIN_VAR = 0.020
# Degenerate-cluster de-bias. When height is unobservable the residual is not
# random, it is a clean predictable vector: offset = body_height * mean_parallax.
# So rather than shrug, subtract it using a nominal detected-figure height.
# Getting that height wrong by 0.3 m leaves only 0.3*|tan beta| of residual,
# still far better than the uncorrected metre-plus error.
ASSUMED_BODY_H_M = 0.55   # nominal height of a detected figure's box centroid
BODY_H_SIGMA_M   = 0.35   # uncertainty in that height -> uncertainty along the bias axis
NADIR_KEEP_FRAC  = 0.30   # fraction of most-nadir detections kept in the degenerate fallback
NADIR_KEEP_MIN   = 8      # ...but never fewer than this many
# Figure -> marker offset. We estimate the detected FIGURE; scoring is against
# the victim's waist/centroid MARKER, which can sit ~1.3 m away for a LYING
# body. Projecting the full box FOOTPRINT (all four corners) and taking its
# centroid lands nearer mid-body than a single centre pixel does, because the
# box's long axis follows the body. Costs nothing and removes a bias the
# pipeline did not model at all.
USE_BOX_FOOTPRINT = True


FORCE_ODOMETRY = False       # redo stage 1 even if camera_poses.csv exists
FORCE_MAPPING  = False        # redo stage 2 even if wall_raw.npy exists
SHOW_PLOTS     = False        # stage 1 matplotlib windows (figure is saved anyway)

# ---- fused stage 1+2 (single pass over the video) ----
FUSE_STAGES  = True           # odometry and mapping in ONE video pass,
                              # one shared YOLO inference per frame
SAVE_FRAMES  = False          # write captured_frames/*.jpg (only needed if
                              # you want to re-run legacy stage 2 separately;
                              # costs ~2 GB and a lot of disk time)
SHOW_ODOM    = True           # stage-1 odometry/trajectory windows
                              # (set False for a further speedup)
DEVICE       = None           # autodetected below: "cuda" if available else "cpu"

# ---- canvas / scale (competition spec) ----
MAP_RES_M   = 0.05
CANVAS      = 600
OUT_SIZE    = 600
OUT_THICK   = 4
OUT_MARGIN  = 24              # unused by the fixed-scale render, kept for API

# ---- STAGE 3a: intensity filter (frozen from the tuner) ----
N_DIRECTIONS  = 2
LOW_FREQ_KEEP = 3
WEDGE_DEG     = 12.9
KEEP_FRACTION = 0.84
INT_WIN       = 3
INT_Q         = 0.44

# ---- STAGE 3b: geometry (frozen from the tuner) ----
CFG = dict(
    hough_threshold=20,
    hough_min_len=10,
    hough_max_gap=5,
    link_eps=23.0,
    lateral_thresh=5.0,
    merge_dist=15.0,
    min_wall_len=24.0,
    snap_tol=40.0,
    door_width=1.24,
    th1=0.10,
    cov_tol=3,
    min_stub_m=0.25,
    max_door_m=2.00,
)

# ---- STAGE 3c: trajectory registration ----
POSE_ORIGIN_PX = (600.0, 600.0)   # pixel of the 1200-grid that world (0,0) maps to
POSE_Y_UP      = True
ALIGN_REFINE   = 6                # +-px local refine, 0 = off

# ---- STAGE 4: ground truth comparison ----
GT_ROT_K        = 1               # our south = original's west
GT_MIRROR       = False
WALL_THICK_M    = 0.2
INCLUDE_KINDS   = ("Wall", "Window")
TOLERANCES_PX   = (0, 1, 2, 4, 8)
SHIFT_SEARCH_PX = 80
THICK_PX        = max(1, int(round(WALL_THICK_M / MAP_RES_M)))
ROT_K, MIRROR   = GT_ROT_K, GT_MIRROR       # names the compare code uses
# =============================================================================

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(TUNING_DIR, exist_ok=True)




# =============================================================================
# ========  STAGE 1  VISUAL ODOMETRY  (final_10_consistent, verbatim)  ========
# =============================================================================
# --- MATHEMATICALLY VERIFIED CAMERA PARAMETERS (Phase 1) ---
Z0 = 96.48            # True Initial Altitude (mm)
FOCAL_LENGTH = 569.84 # True Focal Length (pixels)
FPS = 62.5
MARKER_DISTANCE_MM = 375.0 # True physical distance between ArUco centers
MARKER_DIAG_DISTANCE_MM = 375.0 * np.sqrt(2)  # pair (1,2) is the diagonal

# --- GYRO COMPENSATION CONFIGURATION ---
GYRO_ROLL_SIGN = 1.0  
GYRO_PITCH_SIGN = 1.0 

# --- KINEMATIC GATING (The Experiment) ---
# Maximum allowed discrepancy between IMU expected distance and Camera distance per frame.
# Increased to 40mm to reduce the "parachute" dampening effect while still blocking teleports.
MAX_DEVIATION_MM = 40.0 

# --- FLOOR-ONLY FEATURE TRACKING (the one change vs the original) ---
# Wall/object features at height h produce flow amplified by z/(z-h) vs
# floor features; the median gets captured by them in wall-heavy frames,
# biasing translation. Spawn features on floor only; evict wall-wanderers.
USE_FLOOR_MASK   = True
WALL_SUSPICION_CONF = 0.20   # loose: anything suspected wall is banned
WALL_DILATE_PX   = 10        # safety margin around wall masks (px)
MASK_EVERY_N     = 2         # recompute floor mask every N frames
MIN_SPAWN_FLOOR  = 25        # fewer masked corners than this -> spawn unmasked
MIN_KEEP_POINTS  = 15        # eviction never drops the pool below this

# --- Optical Flow Parameters ---
feature_params = dict(maxCorners=250, qualityLevel=0.07, minDistance=15, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# --- Z SOURCE (consistency fix, learned from final_1 comparison) ---
# The foveated visual scale is the noisiest instrument in the pipeline and
# continuously modulates current_z = the mm-per-pixel ruler. A corridor
# traversed at believed z=2.40 and retraced at z=2.31 yields legs of
# different lengths -> retrace misalignment. The flights hold altitude in
# cruise, so: z comes from the ArUco anchor (absolute, ID-correct) plus the
# IMU vertical channel for genuine climbs. Visual z is disabled.
Z_FROM_VISION = False

# --- Navigation Constraints ---
ROTATION_THRESHOLD = 0.003 # Radians per frame

def load_imu_data(csv_path):
    import pandas as pd
    """Calculates Absolute Heading, Pure Linear Accel (X,Y,Z), and Raw Gyro Rates."""
    df = pd.read_csv(csv_path)
    t = df['timestamp'].values
    if np.max(t) > 1000: t = t / 1000.0
    
    dt = np.diff(t)
    dt = np.insert(dt, 0, 0)
    
    # 1. Compass Heading
    comp_x = df['comp_x'].values
    comp_y = df['comp_y'].values
    yaw_rad = np.arctan2(-comp_x, comp_y)
    
    # 2. Gyroscope Rates (Raw instantaneous, no drifting integration!)
    w_x = df['w_x'].values
    w_y = df['w_y'].values
    
    # 3. Accelerometer processing (Remove gravity and tilt leakage)
    roll = np.cumsum(w_x * dt)
    pitch = np.cumsum(w_y * dt)
    
    ax = df['acc_x'].values
    ay = df['acc_y'].values
    az = df['acc_z'].values
    
    # Gravity removal in local frame based on tilt
    g = 9.81
    g_x = -g * np.sin(pitch)
    g_y = g * np.sin(roll) * np.cos(pitch)
    g_z = g * np.cos(roll) * np.cos(pitch)
    
    ax_linear = ax - g_x
    ay_linear = ay - g_y
    az_linear = az - g_z
    
    return t, yaw_rad, ax_linear, ay_linear, az_linear, w_x, w_y

def aruco_altitude(corners, ids):
    """Absolute altitude from a SPECIFIC ArUco marker pair, selected by ID.
    Pairs (0,1)/(0,2) are 375 mm apart; (1,2) is the 375*sqrt(2) diagonal.
    (The original code used the first two DETECTED markers, which silently
    inflated altitude by sqrt(2) whenever the detector returned pair 1&2.)"""
    centers = {}
    for c, mid in zip(corners, ids.flatten()):
        centers[int(mid)] = np.mean(c[0], axis=0)

    for a, b, dist_mm in ((0, 1, MARKER_DISTANCE_MM),
                          (0, 2, MARKER_DISTANCE_MM),
                          (1, 2, MARKER_DIAG_DISTANCE_MM)):
        if a in centers and b in centers:
            pixel_distance = np.linalg.norm(centers[a] - centers[b])
            if pixel_distance > 5:
                return FOCAL_LENGTH * (dist_mm / pixel_distance)
    return None

def pick_device():
    """cuda if torch sees a GPU, else cpu. Called once, result cached."""
    global DEVICE
    if DEVICE is None:
        try:
            import torch
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            if DEVICE == "cuda":
                print("[gpu] CUDA available: %s"
                      % torch.cuda.get_device_name(0))
            else:
                print("[gpu] no CUDA device, running YOLO on CPU")
        except Exception:
            DEVICE = "cpu"
            print("[gpu] torch not importable, running YOLO on CPU")
    return DEVICE


def load_yolo(path):
    from ultralytics import YOLO
    m = YOLO(path)
    dev = pick_device()
    try:
        m.to(dev)
    except Exception:
        pass
    return m


def infer_masks(model, frame):
    """ONE YOLO inference -> (floor_mask, wall_mask).

    Post-processing is copied verbatim from compute_floor_mask and
    wall_suspicion_mask; both thresholds are 0.20, so the raw union is
    shared and only the morphology differs. This is the fusion that
    halves the model workload.
    """
    h, w = frame.shape[:2]
    res = model(frame, verbose=False, device=DEVICE)[0]
    wall_f = np.zeros((h, w), dtype=np.uint8)      # for the floor mask
    wall_s = np.zeros((h, w), dtype=np.uint8)      # for the seg mask
    if res.masks is not None and res.boxes is not None:
        conf = res.boxes.conf.cpu().numpy()
        keep_f = conf >= WALL_SUSPICION_CONF
        keep_s = conf >= SEG_CONF
        if keep_f.any():
            m = (res.masks.data.cpu().numpy()[keep_f].sum(axis=0) > 0)
            wall_f = cv2.resize(m.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST)
        if keep_s.any():
            m = (res.masks.data.cpu().numpy()[keep_s].sum(axis=0) > 0)
            wall_s = cv2.resize(m.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST)
    if WALL_DILATE_PX > 0:
        wall_f = cv2.dilate(wall_f, np.ones((WALL_DILATE_PX,) * 2, np.uint8))
    floor_mask = ((wall_f == 0) * 255).astype(np.uint8)
    if MASK_OPEN > 1:
        wall_s = cv2.morphologyEx(wall_s, cv2.MORPH_OPEN,
                                  np.ones((MASK_OPEN, MASK_OPEN), np.uint8))
    return floor_mask, wall_s


def compute_floor_mask(model, frame):
    """255 where features are ALLOWED (floor), 0 where banned
    (suspected wall + dilated safety margin)."""
    h, w = frame.shape[:2]
    res = model(frame, verbose=False)[0]
    wall = np.zeros((h, w), dtype=np.uint8)
    if res.masks is not None and res.boxes is not None:
        keep = res.boxes.conf.cpu().numpy() >= WALL_SUSPICION_CONF
        if keep.any():
            m = (res.masks.data.cpu().numpy()[keep].sum(axis=0) > 0)
            wall = cv2.resize(m.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
    if WALL_DILATE_PX > 0:
        wall = cv2.dilate(wall, np.ones((WALL_DILATE_PX,) * 2, np.uint8))
    return ((wall == 0) * 255).astype(np.uint8)

def image_to_global(u, v, drone_pos, theta, orig_w, orig_h):
    """Project a pixel (u, v) of the down-facing flyover camera to global map
    coordinates in millimetres. This is the exact projection tested in
    victims_location.py (pinhole inversion with the Z0 altitude fix)."""
    x_cam, y_cam, z_cam_mm = drone_pos
    h_mm = z_cam_mm                       # altitude already = height above floor
    if h_mm <= 10:
        return None
    uc, vc = orig_w / 2.0, orig_h / 2.0
    x_local = ((u - uc) * h_mm) / FOCAL_LENGTH
    y_local = -((v - vc) * h_mm) / FOCAL_LENGTH
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_global = x_cam + (x_local * cos_t + y_local * sin_t)
    y_global = y_cam + (-x_local * sin_t + y_local * cos_t)
    return (x_global, y_global)


def clamp_f(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _weighted_geometric_median(points, weights, iters=64):
    """Outlier-robust 2D centre (Weiszfeld). Fallback + sanity anchor."""
    P = np.average(points, axis=0, weights=weights)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(points - P, axis=1), 1e-6)
        P_new = np.average(points, axis=0, weights=weights / d)
        if np.linalg.norm(P_new - P) < 1e-7:
            break
        P = P_new
    return P


def _parallax_stats(members):
    """Weighted mean and covariance of the per-view parallax vector
        p_i = (ground projection - camera xy) / altitude = tan(beta) * azimuth.

    Returns (mu, spread_max, spread_min, major_dir). `mu` is the mean parallax:
    the ground projections are displaced from the true position by
    body_height * mu, so mu points ALONG the bias (from the true victim toward
    where the projections land). `spread_max` is the largest variance of p over
    the cluster and is the observability of the target's height: no variation
    means the bias is a constant that cannot be estimated from the data."""
    P, W = [], []
    for m in members:
        C = m["C"]
        H = max(0.1, C[2])
        P.append([(m["x_m"] - C[0]) / H, (m["y_m"] - C[1]) / H])
        W.append(max(1e-6, m["w"]))
    P = np.asarray(P, float)
    W = np.asarray(W, float)
    W = W / W.sum()
    mu = (P * W[:, None]).sum(0)
    D = P - mu
    S = (D * W[:, None]).T @ D
    evals, evecs = np.linalg.eigh(S)
    return (mu, float(max(evals[1], 0.0)), float(max(evals[0], 0.0)),
            evecs[:, 1])


def _triangulate_rays(members):
    """Huber-robust least-squares closest point to a cluster's viewing rays. Each
    detection gives a ray (camera centre C_i, unit dir d_i); P minimises the
    weighted sum of squared perpendicular distances, A P = b with
    A = sum w_i (I - d_i d_i^T). IRLS rejects outliers. Returns (x, y) or None on
    a singular solve / implausible recovered height. Whether this result should
    be TRUSTED is decided by _parallax_stats, not here: the solve can be
    numerically fine and still be physically unobservable."""
    Cs, ds, w0 = [], [], []
    for m in members:
        C = np.asarray(m["C"], float)
        v = np.array([m["x_m"], m["y_m"], 0.0], float) - C
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        Cs.append(C); ds.append(v / n); w0.append(max(1e-3, m["w"]))
    if len(Cs) < 3:
        return None
    Cs, ds, w0 = np.asarray(Cs), np.asarray(ds), np.asarray(w0)
    P = None
    for _ in range(IRLS_ITERS):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for i in range(len(Cs)):
            M = np.eye(3) - np.outer(ds[i], ds[i])
            wi = w0[i]
            if P is not None:
                r = np.linalg.norm(M @ (P - Cs[i]))
                if r > HUBER_DELTA_M:
                    wi *= HUBER_DELTA_M / r
            A += wi * M; b += wi * (M @ Cs[i])
        try:
            P = np.linalg.solve(A + 1e-6 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return None
    if P is None or abs(P[2]) > TRI_MAX_Z_M:
        return None
    return (float(P[0]), float(P[1]))


def _nadir_subset(members):
    """The most-NADIR detections of a cluster: those whose viewing ray is closest
    to straight down. Their ground projections carry the smallest h*tan(beta)
    parallax bias, so when the geometry is degenerate these are the only
    projections worth averaging. Falls back to everything on a small cluster."""
    scored = []
    for m in members:
        C = m["C"]
        d_h = math.hypot(m["x_m"] - C[0], m["y_m"] - C[1])
        H = max(0.1, C[2])
        scored.append((d_h / H, m))             # tan(beta): smaller = more nadir
    scored.sort(key=lambda t: t[0])
    k = max(NADIR_KEEP_MIN, int(round(len(scored) * NADIR_KEEP_FRAC)))
    k = min(k, len(scored))
    return [m for _, m in scored[:k]]


def _robust_victim_location(members):
    """Best (x, y) for a cluster, plus an honest uncertainty for it.

    Enough parallax variation (the drone's viewing geometry genuinely changed
    across the cluster) -> the target's height is observable, so multi-view
    triangulation recovers the true 3D point and the z=0 parallax bias cancels.

    Not enough -> height is unobservable and EVERY view carries the same
    h*tan(beta) offset. Averaging or median-ing the projections cannot remove a
    constant. But that constant is predictable, so we estimate it: take the
    least-biased (most-nadir) views and subtract ASSUMED_BODY_H_M * mean_parallax
    analytically. The leftover uncertainty is the body-height uncertainty
    projected along that same axis, which is exported so the ground robot knows
    which way to sweep.

    Returns (x, y, sigma_major, sigma_minor, weak_ux, weak_uy, quality) with
    quality in {'tri', 'debias', 'median'}."""
    pts = np.array([[m["x_m"], m["y_m"]] for m in members], float)
    w = np.array([m["w"] for m in members], float)
    if w.sum() <= 0:
        w = np.ones(len(members))
    med = _weighted_geometric_median(pts, w)

    mu, spread_max, _spread_min, _major = _parallax_stats(members)
    mu_n = float(np.linalg.norm(mu))
    # Unit vector pointing from the biased projections back toward the truth.
    if mu_n > 1e-6:
        back = (-mu[0] / mu_n, -mu[1] / mu_n)
    else:
        back = (1.0, 0.0)

    observable = spread_max >= PARALLAX_MIN_VAR
    tri = _triangulate_rays(members)

    if observable and tri is not None and \
            math.hypot(tri[0] - med[0], tri[1] - med[1]) <= TRI_TRUST_M:
        # Height solved: residual is dominated by detection noise, not geometry.
        sig = float(clamp_f(0.35 / math.sqrt(max(spread_max, 1e-6)) * 0.05,
                            0.15, 1.20))
        return (tri[0], tri[1], sig, sig * 0.7, back[0], back[1], "tri")

    # Degenerate: de-bias the least-biased views analytically.
    sub = _nadir_subset(members)
    sp = np.array([[m["x_m"], m["y_m"]] for m in sub], float)
    sw = np.array([m["w"] for m in sub], float)
    if sw.sum() <= 0:
        sw = np.ones(len(sub))
    nm = _weighted_geometric_median(sp, sw)
    mu_sub, _sm, _sn, _mj = _parallax_stats(sub)
    x = float(nm[0] - ASSUMED_BODY_H_M * mu_sub[0])
    y = float(nm[1] - ASSUMED_BODY_H_M * mu_sub[1])
    # What is left is our uncertainty in the body height, times the parallax.
    mu_sub_n = float(np.linalg.norm(mu_sub))
    sig_major = float(clamp_f(BODY_H_SIGMA_M * mu_sub_n, 0.25, 2.50))
    if mu_sub_n > 1e-6:
        back = (-mu_sub[0] / mu_sub_n, -mu_sub[1] / mu_sub_n)
    quality = "debias" if tri is not None else "median"
    return (x, y, sig_major, 0.25, back[0], back[1], quality)


def to_origin_marker_frame(x, y):
    """Rotate a point from the pipeline's native frame into the ORIGIN-MARKER
    reference frame the deliverables must use. The offline drone odometry zeroed
    its yaw to compass north rather than the origin marker's +X axis, so the whole
    pipeline map comes out rotated 90 degrees; (x, y) -> (y, -x) rotates it back.

    This MUST be applied before writing victim_location_estimates.csv: the marking
    supervisor adds ONLY the origin translation (coordinate_offset, no rotation)
    before scoring each estimate against the true VICTIM_MARKER world position, so
    an unrotated CSV lands ~90 degrees off in world space and scores ~0 on victim
    localization accuracy (a full half of the 30% Video Information Extraction
    component). The same rotation is applied to wall_estimates.csv so the ground
    controller, which now reads both files without any further rotation, stays in
    one consistent origin-marker frame."""
    return (y, -x)


def project_box_footprint(cx, cy, bx1, by1, bx2, by2, drone_pos, yaw, v_w, v_h):
    """Project the box footprint (centre + 4 corners) to the ground and return the
    centroid in global mm, or None. The centroid follows the body's long axis, so
    for a lying body it lands nearer mid-figure than a single centre pixel."""
    corners = ((cx, cy), (bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2))
    projected = [image_to_global(px, py, drone_pos, yaw, v_w, v_h)
                 for px, py in corners]
    projected = [p for p in projected if p is not None]
    if not projected:
        return None
    return (float(np.mean([p[0] for p in projected])),
            float(np.mean([p[1] for p in projected])))


def finalize_victims(hits):
    """Cluster projected victim hits by spatial proximity, drop clusters seen
    too few times (noise), and write the competition CSV (one 'x,y' per line,
    metres, origin-marker frame, NO header). A richer table with frequency and
    confidence is also saved under src/ for debugging and confidence tuning.
    The contract CSV is always (re)written, empty if nothing was verified."""
    clusters = []
    for h in hits:
        vx, vy, vc = h["x_m"], h["y_m"], h["conf"]
        matched = False
        for c in clusters:
            ax, ay = c["sum_x"] / c["freq"], c["sum_y"] / c["freq"]
            if np.hypot(vx - ax, vy - ay) <= CLUSTER_RADIUS_M:
                c["sum_x"] += vx
                c["sum_y"] += vy
                c["freq"] += 1
                c["max_conf"] = max(c["max_conf"], vc)
                c["members"].append(h)
                matched = True
                break
        if not matched:
            clusters.append(dict(sum_x=vx, sum_y=vy, freq=1, max_conf=vc,
                                 members=[h]))

    # Merge pass: a single victim (especially a large lying one whose ground
    # projections spread out) can seed two separate clusters. Agglomeratively
    # merge clusters whose CENTRES are within VICTIM_MERGE_RADIUS_M, summing
    # frequencies, so one victim yields one estimate. Estimate-count consistency
    # is explicitly part of the 30% Video Info Extraction score, and one victim
    # split in two is the most likely cause of an over-count.
    merged_any = True
    while merged_any and len(clusters) > 1:
        merged_any = False
        for i in range(len(clusters)):
            ci = clusters[i]
            cix, ciy = ci["sum_x"] / ci["freq"], ci["sum_y"] / ci["freq"]
            for j in range(i + 1, len(clusters)):
                cj = clusters[j]
                cjx, cjy = cj["sum_x"] / cj["freq"], cj["sum_y"] / cj["freq"]
                d = float(np.hypot(cix - cjx, ciy - cjy))
                if d <= VICTIM_MERGE_RADIUS_M:
                    print("[victims] MERGE clusters (%.2f,%.2f)+(%.2f,%.2f) "
                          "%.2f m apart, freqs %d+%d -> %d"
                          % (cix, ciy, cjx, cjy, d, ci["freq"], cj["freq"],
                             ci["freq"] + cj["freq"]))
                    ci["sum_x"] += cj["sum_x"]; ci["sum_y"] += cj["sum_y"]
                    ci["freq"] += cj["freq"]
                    ci["max_conf"] = max(ci["max_conf"], cj["max_conf"])
                    ci["members"].extend(cj["members"])
                    clusters.pop(j)
                    merged_any = True
                    break
            if merged_any:
                break

    # Diagnostic: cluster frequency distribution before the noise filter.
    freqs_sorted = sorted((c["freq"] for c in clusters), reverse=True)
    print("[victims] %d clusters after merge, before frequency filter; freqs: %s"
          % (len(clusters), freqs_sorted))

    verified = []
    for c in clusters:
        if c["freq"] < MIN_DETECTION_HITS:
            continue
        rx, ry, smaj, smin, wux, wuy, qual = _robust_victim_location(c["members"])
        mx, my = c["sum_x"] / c["freq"], c["sum_y"] / c["freq"]
        print("[victims]   cluster f=%d: mean=(%.2f,%.2f) -> robust=(%.2f,%.2f) "
              "[%s] parallax correction %.2f m, sigma major %.2f m along "
              "(%+.2f,%+.2f)"
              % (c["freq"], mx, my, rx, ry, qual,
                 np.hypot(rx - mx, ry - my), smaj, wux, wuy))
        if qual in ("debias", "median"):
            print("[victims]     ^ NARROW VIEWING ARC: target height not "
                  "observable, applied h*tan(beta) de-bias; true victim expected "
                  "within ~%.2f m along (%+.2f,%+.2f)" % (smaj, wux, wuy))
        verified.append(dict(x=rx, y=ry, frequency=c["freq"],
                             confidence=c["max_conf"], sigma_major=smaj,
                             sigma_minor=smin, weak_ux=wux, weak_uy=wuy,
                             quality=qual))

    # Optional top-N cap: when the true victim count is known, keep only the N
    # most-seen clusters so we never over-submit (which the scoring penalises).
    if EXPECTED_VICTIM_COUNT is not None and len(verified) > EXPECTED_VICTIM_COUNT:
        verified.sort(key=lambda v: v["frequency"], reverse=True)
        verified = verified[:EXPECTED_VICTIM_COUNT]

    os.makedirs(os.path.dirname(VICTIM_OUT_CSV), exist_ok=True)
    with open(VICTIM_OUT_CSV, "w") as f:
        for v in verified:
            # Deliverable is scored in the origin-marker frame (see
            # to_origin_marker_frame): bake the rotation in here at source.
            mx, my = to_origin_marker_frame(v["x"], v["y"])
            f.write("%.4f,%.4f\n" % (mx, my))
    # Uncertainty sidecar, row-aligned with the contract CSV above. The contract
    # file must stay a bare x,y list, so the ellipse goes here for the ground
    # controller to read (missing file = controller falls back to a fixed radius).
    os.makedirs(os.path.dirname(VICTIM_UNC_CSV), exist_ok=True)
    with open(VICTIM_UNC_CSV, "w") as f:
        f.write("sigma_major_m,sigma_minor_m,weak_ux,weak_uy,quality\n")
        for v in verified:
            f.write("%.4f,%.4f,%.4f,%.4f,%s\n"
                    % (v["sigma_major"], v["sigma_minor"],
                       v["weak_ux"], v["weak_uy"], v["quality"]))
    print("[victims] %d raw detection(s), %d cluster(s), %d verified -> %s"
          % (len(hits), len(clusters), len(verified), VICTIM_OUT_CSV))
    for i, v in enumerate(verified):
        print("[victims]   victim %d: (%.2f, %.2f) m  seen %d frames, "
              "max conf %.2f, %s, sigma_major %.2f m"
              % (i + 1, v["x"], v["y"], v["frequency"], v["confidence"],
                 v["quality"], v["sigma_major"]))

    if verified:
        dbg = os.path.join(MAPPING_DIR, "victim_locations.csv")
        with open(dbg, "w") as f:
            f.write("x_m,y_m,frequency,confidence,sigma_major_m,sigma_minor_m,"
                    "weak_ux,weak_uy,quality\n")
            for v in verified:
                f.write("%.4f,%.4f,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%s\n"
                        % (v["x"], v["y"], v["frequency"], v["confidence"],
                           v["sigma_major"], v["sigma_minor"],
                           v["weak_ux"], v["weak_uy"], v["quality"]))


def run_odometry(video_path, csv_path, mapper=None):
    """STAGE 1, and when mapper is given also STAGE 2, in one video pass.
    One shared YOLO inference per frame feeds BOTH the floor mask (feature
    eviction) and the wall mask (mapping)."""
    import pandas as pd
    import matplotlib.pyplot as plt
    print("Loading IMU Data...")
    imu_t, imu_yaw, imu_ax, imu_ay, imu_az, imu_wx, imu_wy = load_imu_data(csv_path)
    
    # --- Floor-mask model (shared with the mapper when fused) ---
    floor_model = None
    if USE_FLOOR_MASK or mapper is not None:
        try:
            print("Loading wall model (shared floor + wall masks)...")
            floor_model = load_yolo(MODEL_PATH)
        except Exception as e:
            print(f"WARNING: floor mask disabled ({e}) — running with "
                  f"unmasked features like the original.")
            if mapper is not None:
                raise SystemExit("fused mapping needs the YOLO model") from e

    # --- Victim detector (runs in this same pass; nothing saved to disk) ---
    victim_model = None
    if DETECT_VICTIMS:
        try:
            print(f"Loading victim detector ({os.path.basename(VICTIM_MODEL_PATH)})...")
            victim_model = load_yolo(VICTIM_MODEL_PATH)
        except Exception as e:
            print(f"WARNING: victim detection disabled ({e}).")
            victim_model = None

    # --- Setup ArUco Detector for Phase Switching ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters_create()
        detector = None
    
    cap = cv2.VideoCapture(video_path)
    
    # Global Position Variables
    global_x = 0.0 
    global_y = 0.0  
    current_z = Z0  
    
    # Velocity integrators
    current_vz = 0.0 
    vx_imu = 0.0
    vy_imu = 0.0
    
    trajectory_x = []
    trajectory_y = []
    altitude_log = []
    pose_log = [] # Array to store mapping telemetry
    center_colors = [] # Array to store dominant floor colors
    victim_hits = [] # projected victim detections, clustered after the pass
    
    # --- Live 2D Map Setup ---
    map_size = 800
    map_scale = 0.03 
    map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    map_center_x, map_center_y = map_size // 2, map_size // 2
    
    cv2.line(map_img, (map_center_x, 0), (map_center_x, map_size), (50, 50, 50), 1)
    cv2.line(map_img, (0, map_center_y), (map_size, map_center_y), (50, 50, 50), 1)
    
    cv2.putText(map_img, "N", (map_center_x - 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(map_img, "S", (map_center_x - 10, map_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(map_img, "E", (map_size - 25, map_center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(map_img, "W", (5, map_center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    prev_map_x = int(global_x * map_scale + map_center_x)
    prev_map_y = int(-global_y * map_scale + map_center_y) 
    
    old_gray = None
    old_points = None
    prev_yaw = None
    prev_az = None
    frame_idx = 0
    dt = 1.0 / FPS
    
    floor_mask = None
    wall_mask_cache = (-1, None)     # (frame_idx it belongs to, wall mask)
    saved_idx = 0                    # pose rows written so far (mapper stride)
    
    # State Machine Variables
    flight_phase = "TAKEOFF (MARKERS VISIBLE)"
    missing_marker_frames = 0
    
    print("\nStarting Visual Odometry Pipeline (Foveated Scale + Kinematic Bounding + Floor-Masked Features)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- PRISTINE FRAME FOR MAPPING ---
        # Save a clean copy BEFORE any optical flow dots or UI text are drawn!
        clean_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- FLOOR MASK (recomputed every N frames) ---
        # One inference produces BOTH masks; the wall mask is cached for
        # the mapper so a frame is never inferred twice.
        if floor_model is not None and (floor_mask is None or
                                        frame_idx % MASK_EVERY_N == 0):
            floor_mask, _wm = infer_masks(floor_model, clean_frame)
            wall_mask_cache = (frame_idx, _wm)
        
        # --- PHASE DETECTION (ArUco Trigger) ---
        if detector:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
        if ids is not None:
            missing_marker_frames = 0
            flight_phase = "TAKEOFF (MARKERS VISIBLE)"
            
            # --- ROBUST UPGRADE: ABSOLUTE ALTITUDE (ID-selected pair) ---
            z_marker = aruco_altitude(corners, ids)
            if z_marker is not None:
                current_z = z_marker
                
                if len(altitude_log) > 0:
                    current_vz = ((current_z - altitude_log[-1]) / 1000.0) / dt
        else:
            missing_marker_frames += 1
            if missing_marker_frames > 15:
                flight_phase = "CRUISE (RATE-OF-CHANGE LOCK)"
        
        # Time Sync
        current_time = frame_idx / FPS
        idx = (np.abs(imu_t - current_time)).argmin()
        
        current_yaw = imu_yaw[idx]
        current_ax = imu_ax[idx]
        current_ay = imu_ay[idx]
        current_az = imu_az[idx]
        current_wx = imu_wx[idx]
        current_wy = imu_wy[idx]
        
        if prev_yaw is None: prev_yaw = current_yaw
        if prev_az is None: prev_az = current_az
            
        # --- IMU KINEMATIC INTEGRATION ---
        rate_of_change_az = abs(current_az - prev_az)
        is_vertically_active = (rate_of_change_az > 0.05) or (abs(current_az) > 0.3)
        
        if flight_phase == "TAKEOFF (MARKERS VISIBLE)":
            z_imu_pred = current_z 
            vx_imu = 0.0
            vy_imu = 0.0
        else:
            if is_vertically_active:
                current_vz += current_az * dt
                current_vz *= 0.98 
            else:
                current_vz = 0.0
                
            delta_z_imu = current_vz * dt * 1000.0 
            z_imu_pred = current_z + delta_z_imu
            
            # Update Horizontal IMU Velocity (with relaxed decay to allow momentum)
            vx_imu = (vx_imu + current_ax * dt) * 0.995
            vy_imu = (vy_imu + current_ay * dt) * 0.995
        
        is_rotating = False
            
        # --- 1. Tracking Phase ---
        if old_gray is not None and old_points is not None and len(old_points) > 0:
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_points, None, **lk_params)
            
            good_new = new_points[status == 1]
            good_old = old_points[status == 1]
            
            # --- EVICT FEATURES THAT WANDERED ONTO WALLS ---
            # (never below MIN_KEEP_POINTS: a few wall stragglers beat an
            # empty pool)
            if floor_mask is not None and len(good_new) > MIN_KEEP_POINTS:
                fx = np.clip(good_new[:, 0].astype(int), 0, floor_mask.shape[1] - 1)
                fy = np.clip(good_new[:, 1].astype(int), 0, floor_mask.shape[0] - 1)
                on_floor = floor_mask[fy, fx] > 0
                if on_floor.sum() >= MIN_KEEP_POINTS:
                    good_new = good_new[on_floor]
                    good_old = good_old[on_floor]
            
            if len(good_new) > 5:
                # --- RAW PIXEL MOVEMENT ---
                raw_dx_px = np.median(good_new[:, 0] - good_old[:, 0])
                raw_dy_px = np.median(good_new[:, 1] - good_old[:, 1])
                
                # --- ANALYTICAL GYRO COMPENSATION ---
                tilt_dx_px = (current_wx * dt * FOCAL_LENGTH) * GYRO_ROLL_SIGN
                tilt_dy_px = (current_wy * dt * FOCAL_LENGTH) * GYRO_PITCH_SIGN
                
                dx_px_total = raw_dx_px - tilt_dx_px
                dy_px_total = raw_dy_px - tilt_dy_px
                pixel_speed = np.sqrt(dx_px_total**2 + dy_px_total**2)
                
                # --- CENTER-FOVEATED ALTITUDE SCALING ---
                cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
                dist_to_center = np.linalg.norm(good_old - [cx, cy], axis=1)
                
                center_mask = dist_to_center < (min(cx, cy) * 0.4) 
                valid_old_z = good_old[center_mask]
                valid_new_z = good_new[center_mask]
                
                scale_ratio = 1.0
                valid_idx = np.array([])
                if len(valid_old_z) > 5:
                    centroid_old = np.mean(valid_old_z, axis=0)
                    centroid_new = np.mean(valid_new_z, axis=0)
                    dist_old_z = np.linalg.norm(valid_old_z - centroid_old, axis=1)
                    dist_new_z = np.linalg.norm(valid_new_z - centroid_new, axis=1)
                    
                    valid_idx = (dist_new_z > 1.0) & (dist_old_z > 1.0)
                    if np.sum(valid_idx) > 3:
                        scale_ratio = np.median(dist_old_z[valid_idx] / dist_new_z[valid_idx])
                        scale_ratio = np.clip(scale_ratio, 0.98, 1.02) 
                
                z_vis_meas = current_z * scale_ratio
                
                # --- Z-AXIS SENSOR FUSION ---
                if not Z_FROM_VISION:
                    # Consistent ruler: ArUco anchor + IMU vertical only.
                    if flight_phase != "TAKEOFF (MARKERS VISIBLE)":
                        current_z = z_imu_pred
                elif flight_phase == "TAKEOFF (MARKERS VISIBLE)":
                    pass 
                elif flight_phase == "CRUISE (RATE-OF-CHANGE LOCK)" and not is_vertically_active and pixel_speed > 1.5:
                    current_z = z_imu_pred
                elif pixel_speed > 1.5 or np.sum(valid_idx) <= 3:
                    current_z = (0.99 * z_imu_pred) + (0.01 * z_vis_meas)
                else:
                    current_z = (0.80 * z_imu_pred) + (0.20 * z_vis_meas)
                
                # --- TRANSLATION CALCULATIONS ---
                current_scale = FOCAL_LENGTH / current_z 
                
                delta_yaw = current_yaw - prev_yaw
                delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
                is_rotating = abs(delta_yaw) > ROTATION_THRESHOLD
                
                if not is_rotating:
                    # 1. Unbounded Visual Distance
                    dx_local_mm = -(dx_px_total / current_scale)
                    dy_local_mm = (dy_px_total / current_scale) 
                    
                    # 2. Expected IMU Distance
                    expected_dx_mm = vx_imu * dt * 1000.0
                    expected_dy_mm = vy_imu * dt * 1000.0
                    
                    # 3. KINEMATIC GATING (The Bounding Experiment)
                    # We check if the camera's measurement radically deviates from what physics dictates.
                    # Note: You may need to flip signs depending on Webots IMU vs Camera axis layout.
                    dx_error = dx_local_mm - expected_dx_mm
                    dy_error = dy_local_mm - expected_dy_mm
                    
                    if abs(dx_error) > MAX_DEVIATION_MM:
                        dx_local_mm = expected_dx_mm + (np.sign(dx_error) * MAX_DEVIATION_MM)
                        
                    if abs(dy_error) > MAX_DEVIATION_MM:
                        dy_local_mm = expected_dy_mm + (np.sign(dy_error) * MAX_DEVIATION_MM)

                    # 4. Global Rotation
                    dx_global_mm = dx_local_mm * np.cos(current_yaw) + dy_local_mm * np.sin(current_yaw)
                    dy_global_mm = -dx_local_mm * np.sin(current_yaw) + dy_local_mm * np.cos(current_yaw)
                    
                    global_x += dx_global_mm
                    global_y += dy_global_mm
                
                # Draw optical flow tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    color = (0, 165, 255) if is_rotating else (0, 255, 0)
                    cv2.line(frame, (int(a), int(b)), (int(c), int(d)), color, 2)
                    cv2.circle(frame, (int(a), int(b)), 3, color, -1)
            
            old_points = good_new.reshape(-1, 1, 2)
            
        # --- 2. Feature Replenishment Phase (FLOOR FIRST, FALLBACK UNMASKED) ---
        if old_points is None or len(old_points) < 50:
            new_features = cv2.goodFeaturesToTrack(gray, mask=floor_mask, **feature_params)
            if (new_features is None or len(new_features) < MIN_SPAWN_FLOOR) \
                    and floor_mask is not None:
                # Floor too small in this view (corridor) — spawn unmasked
                # rather than track nothing.
                new_features = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if new_features is not None:
                if old_points is not None and len(old_points) > 0:
                    old_points = np.vstack((old_points, new_features))
                else:
                    old_points = new_features
                
                for pt in new_features:
                    cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 4, (0, 0, 255), -1)
            
        old_gray = gray.copy()
        prev_yaw = current_yaw 
        prev_az = current_az
        
        trajectory_x.append(global_x)
        trajectory_y.append(global_y)
        altitude_log.append(current_z)

        # --- RECORD TELEMETRY & FRAMES FOR MAPPING ---
        # Save frames and pose logs strictly when not rotating to prevent motion blur in walls5.py
        if not is_rotating:
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(FRAMES_DIR, frame_filename)
            
            # Commit the pristine, spotless frame to disk! (No green dots)
            if SAVE_FRAMES:
                cv2.imwrite(frame_path, clean_frame)
            
            # Store critical pose elements needed to compute the projection matrix
            pose_row = {
                'frame_file': frame_filename,
                'global_x_mm': global_x,
                'global_y_mm': global_y,
                'altitude_z_mm': current_z,
                'yaw_rad': current_yaw
            }
            pose_log.append(pose_row)

            # --- FUSED VICTIM DETECTION: same frame, same pose ---
            # Runs on every non-rotating frame, matching the cadence
            # victims_location.py used (so MIN_DETECTION_HITS stays comparable).
            if victim_model is not None:
                v_h, v_w = clean_frame.shape[:2]
                drone_pos = (global_x, global_y, current_z)
                v_results = victim_model.predict(source=clean_frame,
                                                 classes=VICTIM_CLASSES,
                                                 conf=VICTIM_CONF, verbose=False)
                for v_res in v_results:
                    for box in v_res.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        if USE_BOX_FOOTPRINT:
                            gp = project_box_footprint(cx, cy, bx1, by1, bx2, by2,
                                                       drone_pos, current_yaw,
                                                       v_w, v_h)
                        else:
                            gp = image_to_global(cx, cy, drone_pos, current_yaw,
                                                 v_w, v_h)
                        if gp is not None:
                            gx_m, gy_m = gp[0] / 1000.0, gp[1] / 1000.0
                            # Parked ROSbots on the launch pad read as a "person"
                            # from directly overhead; never a real victim.
                            if np.hypot(gx_m, gy_m) < LAUNCH_EXCLUDE_M:
                                continue
                            vconf = float(box.conf[0].cpu().numpy())
                            C_m = (drone_pos[0] / 1000.0, drone_pos[1] / 1000.0,
                                   drone_pos[2] / 1000.0)
                            d_h = np.hypot(gx_m - C_m[0], gy_m - C_m[1])
                            H_m = max(0.1, C_m[2])
                            # cos^2 of the off-nadir angle: weight nadir views most.
                            nadir_w = (H_m * H_m) / (H_m * H_m + d_h * d_h)
                            victim_hits.append(dict(x_m=gx_m, y_m=gy_m, conf=vconf,
                                                    C=C_m, w=vconf * nadir_w))

            # --- FUSED MAPPING: feed this pose row straight to the mapper ---
            if mapper is not None:
                wm = None
                if saved_idx % FRAME_STRIDE == 0:
                    if wall_mask_cache[0] == frame_idx:
                        wm = wall_mask_cache[1]      # reuse this frame's inference
                    else:
                        _fm, wm = infer_masks(floor_model, clean_frame)
                mapper.feed(clean_frame if wm is not None else None,
                            pose_row, wm)
                if mapper.quit:
                    break
            saved_idx += 1
            
            # --- Sample Center Color for K-Means Clustering ---
            if frame_idx % 15 == 0:  # Sample every 15th frame
                h_f, w_f = clean_frame.shape[:2]
                cx, cy = w_f // 2, h_f // 2
                
                # Extract small patch from the clean frame, apply fast blur, and convert to LAB
                patch = clean_frame[cy-10:cy+10, cx-10:cx+10]
                blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
                lab_patch = cv2.cvtColor(blurred_patch, cv2.COLOR_BGR2LAB)
                
                avg_color = np.mean(lab_patch, axis=(0, 1))
                center_colors.append(avg_color)

        # --- Update Live Maps & UI ---
        curr_map_x = int(global_x * map_scale + map_center_x)
        curr_map_y = int(-global_y * map_scale + map_center_y) 
        
        cv2.line(map_img, (prev_map_x, prev_map_y), (curr_map_x, curr_map_y), (255, 0, 0), 2)
        prev_map_x, prev_map_y = curr_map_x, curr_map_y
        
        display_map = map_img.copy()
        
        arrow_len = 30
        heading_x = int(curr_map_x + arrow_len * np.sin(current_yaw))
        heading_y = int(curr_map_y - arrow_len * np.cos(current_yaw))
        cv2.arrowedLine(display_map, (curr_map_x, curr_map_y), (heading_x, heading_y), (0, 0, 255), 2, tipLength=0.3)
        
        cv2.circle(display_map, (curr_map_x, curr_map_y), 6, (0, 255, 0), -1)
        cv2.putText(display_map, f"Pos: {global_x/1000:.2f}m, {global_y/1000:.2f}m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Phase: {flight_phase}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)
        cv2.putText(frame, f"Fused Alt: {current_z/1000:.3f} m", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Pos X: {global_x/1000:.2f} m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Pos Y: {global_y/1000:.2f} m", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if is_rotating:
            cv2.putText(frame, "ROTATION DETECTED: Pausing Translation", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        if SHOW_ODOM:
            cv2.imshow("Drone Visual Odometry", frame)
            cv2.imshow("Live 2D Trajectory", display_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # --- FUSED MAPPING: flush submaps, sweep footprint, save wall_raw.npy ---
    if mapper is not None:
        mapper.finish()

    # --- SAVE MAPPING TELEMETRY ---
    print("\nExporting dedicated mapping telemetry registry to 'mapping_data/camera_poses.csv'...")
    poses_df = pd.DataFrame(pose_log)
    poses_df.to_csv(CSV_PATH, index=False)

    # --- FUSED VICTIM DETECTION: cluster the pass's hits, write the CSV ---
    if DETECT_VICTIMS:
        finalize_victims(victim_hits)

    # --- COMPUTE AND SAVE DOMINANT COLOR PALETTES ---
    if len(center_colors) > 0:
        print("\nComputing dominant floor/table color palettes (K=5)...")
        center_colors_array = np.float32(center_colors)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(center_colors_array, 7, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Save to numpy format so the edge detector can load it instantly
        np.save(os.path.join(MAPPING_DIR, 'dominant_palettes.npy'), centers)
        print("Saved dominant palettes to %s"
              % os.path.join(MAPPING_DIR, 'dominant_palettes.npy'))
        for i, c in enumerate(centers):
            print(f"Palette {i+1} (LAB): {c}")

    if trajectory_x and trajectory_y:
        print("\nGenerating verification maps...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        traj_x_m = [x / 1000.0 for x in trajectory_x]
        traj_y_m = [y / 1000.0 for y in trajectory_y]
        
        ax1.plot(traj_x_m, traj_y_m, label='Drone Trajectory', color='blue', linewidth=2)
        ax1.scatter(traj_x_m[0], traj_y_m[0], color='green', s=100, label='Start Position', zorder=5)
        ax1.scatter(traj_x_m[-1], traj_y_m[-1], color='red', s=100, label='End Position', zorder=5)
        ax1.set_title('2D Drone Flight Path')
        ax1.set_xlabel('Global East (meters)')
        ax1.set_ylabel('Global North (meters)')
        ax1.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.85), arrowprops=dict(facecolor='black', width=3, headwidth=10), ha='center', va='center', fontsize=14, fontweight='bold', xycoords='axes fraction', textcoords='axes fraction')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.axvline(0, color='black', linewidth=1)
        ax1.legend(loc='lower right')
        ax1.axis('equal') 
        
        alt_m = [z / 1000.0 for z in altitude_log]
        time_s = [i / FPS for i in range(len(alt_m))]
        
        ax2.plot(time_s, alt_m, label='IMU/Visual Fused Altitude', color='purple', linewidth=2)
        ax2.set_title('Z-Axis Altitude Verification')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Altitude (meters)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        fig.savefig(os.path.join(MAPPING_DIR, "trajectory_verification.png"),
                    dpi=120)
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)



# =============================================================================
# ====  STAGE 2  WALL GRID FROM FRAMES  (final_segmentation_v23, verbatim)  ===
# =============================================================================
"""
Wall mapper v22 — v21 + TRAJECTORY-GATED GAP EXTENSION (occlusion bridging).

The user's rule, implemented literally:
  - A ray's first floor->wall transition stamps that cell as WALL,
    immediately and permanently. No weights, no log-odds, no minimum hit
    counts, no thresholds. The live map shows it the same frame.
  - Free space is tracked only for visualization/diagnostics; it can NEVER
    erase a wall cell.
  - ALL noise handling happens in post-processing, structurally:
    connect dotted hits (small close) -> keep components that are LINE-LIKE
    (long & thin) or CONNECTED to other kept walls -> drop confetti ->
    optional Hough + dominant-axis snap -> 4 px walls -> centered 600x600.
  - ANNULUS SNAP (v18 fix): snap only applies to hits landing at
    SNAP_MIN_CELLS..SNAP_CELLS from an existing wall (genuine drift
    duplicates). Hits CLOSER than SNAP_MIN_CELLS stamp themselves — in v17
    the full-disc snap teleported gap-filling hits onto existing dots, so
    the first frame's dot pattern fossilized into permanent dashed lines.
    The annulus lets gaps along a wall fill while still merging duplicates.

Keys (SHOW_LIVE): q quit(+save) | p pause | s snapshot | f follow/overview
Tuning layers in mapping_data/tuning/.

SUBMAPS (new in v19, from GPS-denied mapping research): rays stamp into a
small SUBMAP covering ~SUBMAP_ROWS pose rows — short enough that odometry
drift within it stays below one map cell, so each submap is internally
consistent. On completion the submap is aligned to the global map by a
translation-only image correlation (heading is compass-absolute): slide the
submap's wall image over the global wall image within +-SEARCH_CELLS and
maximize wall-on-wall overlap (global dilated 1 cell for tolerance). The
winning offset composites the submap in and becomes the prior for the next
submap, so the correction TRACKS drift. A submap with too little wall
content, or too little overlap with the global map, pastes at the carried
offset — no wild guesses. Re-seen walls slide onto their originals: one
wall in the output. This replaces per-frame scan matching (too sparse) and
feature loop closure (repetitive floor texture) with whole-structure
alignment at the right granularity.
"""

# ============================= CONFIG ========================================

# SEG_Z0 SUBTRACTION LIKELY A BUG: aruco altitude z = f*375/d is ALREADY the
# height above the mat (= floor). Subtracting SEG_Z0 again shortens every
# projected range by ~4% (96/2400), pulling walls toward the drone from
# every viewing side -> ~25 cm doubling at 3 m range, while odometry travel
# (which never subtracts SEG_Z0) stays correct. Set 0.0 to test; restore 96.48
# to reproduce old behavior.
SEG_Z0           = 0.0
FOCAL_LENGTH = 569.84
SEG_CONF     = 0.20       # wall mask confidence

FRAME_STRIDE = 2
N_RAYS       = 900        # angular resolution (0.4 deg)
RAY_STEP_PX  = 3          # radial march step

MAP_RES_M    = 0.05
WORK_PX      = 1200
OUT_PX       = 600
WALL_THICK_PX = 4

# --- Snap-to-wall (loop-closure duplicates) ---
SNAP_ON      = True
SNAP_CELLS   = 6          # snap hits within 0.30 m of existing wall onto it
SNAP_MIN_CELLS = 3        # ...but only if at least this far away: closer
                          # hits stamp themselves (fill gaps along the wall)

# --- Submap architecture ---
SUBMAP_ROWS   = 350       # pose rows per submap (drift within < 1 cell)
SEARCH_CELLS  = 12        # +-0.6 m alignment search around carried offset
MIN_SUB_WALL  = 150       # submap wall cells needed to attempt alignment
MIN_OVERLAP   = 60        # aligned overlap cells needed to accept an offset
STEP_CAP      = 6         # max offset change between consecutive submaps
SNAP_REFRESH = 20         # rebuild nearest-wall lookup every N frames

# --- Post-processing (the ONLY noise filter) ---
POST_CLOSE     = 3        # connect dotted hits into lines
MIN_SEG_LEN_PX = 6        # keep standalone segments >= 0.30 m long ...
MAX_SEG_WIDTH_PX = 8      # ... and <= 0.40 m wide (long-and-thin test)
MIN_CC_CELLS   = 4        # drop components smaller than this outright

# --- Contradiction filter (non-temporal noise removal) ---
# A component is noise if the free layer CONTRADICTS it: rays from other
# frames passed through its cells seeing floor. Real walls — even ones
# stamped in a single frame — are never floor-crossed (rays hit them or are
# occluded; the carve margin keeps near-miss floor votes off the band).
# Intensity/hit-counts are never used, so once-seen walls survive.
CONTRA_FRAC    = 0.6      # component dropped if > this fraction of its
                          # cells were also carved as observed floor
MASK_OPEN      = 3        # despeckle the segmentation mask before rays
                          # (1-2 px flecks never generate hits at all)

# --- Rectilinear pruning (furniture-dent removal) ---
# Furniture hugging a wall occludes the base; rays hit the wall's UPPER
# part above it, which projects overshot -> a bump on the straight line.
# Fix: directional opening in the dominant-axis frame — a cell survives
# only if it belongs to a straight run >= RECT_LEN cells along the
# horizontal OR vertical axis. Long walls trivially qualify; furniture-
# shaped bumps (short both ways) are shaved flush. Prune-only: never adds.
RECT_PRUNE     = True     # set False if it shaves nothing useful for you
RECT_LEN       = 9        # 0.45 m minimum straight run to keep a cell

# --- Trajectory-gated gap extension (occlusion bridging) ---
# Furniture hugging walls occludes the base -> gaps in straight wall runs.
# Fix: extend each fitted wall segment from its endpoints along its own
# direction; FILL the run only if it terminates on another wall within
# MAX_EXTEND_CELLS. ABORT (leave the gap) if the march touches the drone's
# flight corridor — the drone flew through every doorway, so doorways are
# certified not-wall and can never be sealed; furniture gaps are places the
# drone could not fly. The user's constraint, verbatim.
# --- DRONE FOOTPRINT CLEARING (physical presence = certified free) ---
# A wall cannot exist where the drone itself flew. Every frame, a square of
# half-width FOOT_CELLS around the drone's grid cell is cleared of wall
# stamps and LOCKED as free (noise can never restamp it). Tunable live via
# the 'Footprint' trackbar; the final value also sweeps the whole
# trajectory at the end, so earlier frames get the same treatment.
FOOT_CELLS     = 5     # half-width in cells (5 = 0.55 m square)
FOOT_MAX       = 20    # trackbar range

GAP_EXTEND     = True
MAX_EXTEND_CELLS = 50     # bridge at most 2.5 m
TRAJ_CLEAR_CELLS = 4      # drone corridor half-width (0.20 m)
USE_LINE_FIT   = True     # Hough + dominant-axis snap at the end
SNAP_DEG       = 12
BRIDGE_PX      = 15

SHOW_LIVE    = True
VIS_CAM_W    = 860
VIS_VIEW_PX  = 520
VIS_ZOOM     = 1.3

# ============================= GEOMETRY ======================================
def project_ground(us, vs, pose, w, h):
    x_cam, y_cam, z_cam, theta = pose
    H_mm = z_cam - SEG_Z0
    if H_mm <= 10:
        return None, None
    uc, vc = w / 2.0, h / 2.0
    x_loc = ((us - uc) * H_mm) / FOCAL_LENGTH
    y_loc = -((vs - vc) * H_mm) / FOCAL_LENGTH
    c, s = np.cos(theta), np.sin(theta)
    xg = x_cam + (x_loc * c + y_loc * s)
    yg = y_cam + (-x_loc * s + y_loc * c)
    return xg / 1000.0, yg / 1000.0


def world_to_grid(x_m, y_m):
    col = np.round(np.asarray(x_m) / MAP_RES_M + WORK_PX / 2.0).astype(np.int64)
    row = np.round(WORK_PX / 2.0 - np.asarray(y_m) / MAP_RES_M).astype(np.int64)
    return row, col


# ============================= PERCEPTION ====================================
def wall_suspicion_mask(model, frame):
    h, w = frame.shape[:2]
    res = model(frame, verbose=False)[0]
    if res.masks is None or res.boxes is None:
        return np.zeros((h, w), dtype=np.uint8)
    keep = res.boxes.conf.cpu().numpy() >= SEG_CONF
    if not keep.any():
        return np.zeros((h, w), dtype=np.uint8)
    m = (res.masks.data.cpu().numpy()[keep].sum(axis=0) > 0).astype(np.uint8)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    if MASK_OPEN > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             np.ones((MASK_OPEN, MASK_OPEN), np.uint8))
    return m


# ============================= RAY ENGINE ====================================
def ray_observations(mask):
    """First floor->wall transition per ray = hit. Free samples = everything
    crossed before the hit. No weights — a hit is a hit."""
    h, w = mask.shape
    uc, vc = w / 2.0, h / 2.0
    max_r = np.hypot(uc, vc)

    angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    radii = np.arange(0, max_r, RAY_STEP_PX)

    us = uc + radii[None, :] * np.cos(angles)[:, None]
    vs = vc + radii[None, :] * np.sin(angles)[:, None]
    inside = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    ui = np.clip(us.astype(np.int32), 0, w - 1)
    vi = np.clip(vs.astype(np.int32), 0, h - 1)
    samples = mask[vi, ui].astype(np.int8)
    samples[~inside] = -1

    free_u, free_v = [], []
    hit_u, hit_v = [], []

    for ray in range(N_RAYS):
        s = samples[ray]
        exited = s[0] != 1
        for k in range(len(s)):
            if s[k] == -1:
                break
            if not exited:
                if s[k] == 0:
                    exited = True
                continue
            if s[k] == 0:
                free_u.append(us[ray, k])
                free_v.append(vs[ray, k])
            else:
                hit_u.append(us[ray, k])
                hit_v.append(vs[ray, k])
                break  # occluded beyond
    return (np.array(free_u), np.array(free_v),
            np.array(hit_u), np.array(hit_v))


# ============================= SNAP LOOKUP ===================================
def build_snap_lookup(wall):
    if not wall.any():
        return None
    src_img = np.where(wall, 0, 255).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        src_img, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    coords = np.argwhere(wall)   # row-major, matches DIST_LABEL_PIXEL order
    return dist, labels, coords


# ============================= SUBMAP ALIGNMENT ==============================
def align_submap(global_wall, sub_wall, prior_dr, prior_dc):
    """Translation-only correlation of the submap wall image against the
    (1-cell dilated) global wall image, searched around the carried prior.
    Returns (dr, dc, accepted)."""
    ys, xs = np.nonzero(sub_wall)
    if ys.size < MIN_SUB_WALL or not global_wall.any():
        return prior_dr, prior_dc, False
    gdil = cv2.dilate(global_wall.astype(np.uint8),
                      np.ones((3, 3), np.uint8)).astype(bool)
    best_score, best = -1, (prior_dr, prior_dc)
    for dr in range(prior_dr - SEARCH_CELLS, prior_dr + SEARCH_CELLS + 1):
        r = ys + dr
        rok = (r >= 0) & (r < WORK_PX)
        for dc in range(prior_dc - SEARCH_CELLS, prior_dc + SEARCH_CELLS + 1):
            c = xs + dc
            ok = rok & (c >= 0) & (c < WORK_PX)
            if not ok.any():
                continue
            score = int(gdil[r[ok], c[ok]].sum())
            if dr == prior_dr and dc == prior_dc:
                score = int(score * 1.03)      # stay-put tie break
            if score > best_score:
                best_score, best = score, (dr, dc)
    if best_score < MIN_OVERLAP:
        return prior_dr, prior_dc, False
    dr = int(np.clip(best[0], prior_dr - STEP_CAP, prior_dr + STEP_CAP))
    dc = int(np.clip(best[1], prior_dc - STEP_CAP, prior_dc + STEP_CAP))
    return dr, dc, True


def composite(global_wall, global_free, sub_wall, sub_free, dr, dc):
    ys, xs = np.nonzero(sub_wall)
    r, c = ys + dr, xs + dc
    ok = (r >= 0) & (r < WORK_PX) & (c >= 0) & (c < WORK_PX)
    global_wall[r[ok], c[ok]] = True
    ys, xs = np.nonzero(sub_free)
    r, c = ys + dr, xs + dc
    ok = (r >= 0) & (r < WORK_PX) & (c >= 0) & (c < WORK_PX)
    global_free[r[ok], c[ok]] = True


# ============================= ACCUMULATE ====================================
def accumulate(wall, seen_free, mask, pose, w, h, snap=None):
    fu, fv, hu, hv = ray_observations(mask)

    if hu.size:
        xg, yg = project_ground(hu, hv, pose, w, h)
        if xg is not None:
            r, c = world_to_grid(xg, yg)
            ok = (r >= 0) & (r < WORK_PX) & (c >= 0) & (c < WORK_PX)
            r, c = r[ok], c[ok]
            if SNAP_ON and snap is not None:
                dist, labels, coords = snap
                d = dist[r, c]
                near = (d >= SNAP_MIN_CELLS) & (d <= SNAP_CELLS)
                if near.any():
                    lbl = labels[r[near], c[near]] - 1
                    tgt = coords[lbl]
                    r[near] = tgt[:, 0]
                    c[near] = tgt[:, 1]
            wall[r, c] = True          # DIRECT, PERMANENT

    if fu.size:
        xg, yg = project_ground(fu, fv, pose, w, h)
        if xg is not None:
            r, c = world_to_grid(xg, yg)
            ok = (r >= 0) & (r < WORK_PX) & (c >= 0) & (c < WORK_PX)
            seen_free[r[ok], c[ok]] = True   # visualization only
    return True


# ============================= POST-PROCESSING ===============================
def clean_walls(wall, seen_free=None, traj_rc=None):
    """Noise filters: (1) contradiction — components majority-crossed as
    floor by other rays are noise, independent of how often they were
    stamped; (2) structural continuity."""
    raw = (wall.astype(np.uint8)) * 255
    cv2.imwrite(os.path.join(TUNING_DIR, "1_raw_stamps.png"), raw)

    if seen_free is not None:
        n, labels = cv2.connectedComponents(raw)
        killed = np.zeros_like(wall)
        n_kill = 0
        for i in range(1, n):
            comp = labels == i
            frac = float(seen_free[comp].mean())
            if frac > CONTRA_FRAC:
                killed |= comp
                n_kill += 1
        if n_kill:
            wall = wall & ~killed
            raw = (wall.astype(np.uint8)) * 255
            print(f"[contra] dropped {n_kill} floor-contradicted "
                  f"component(s), {int(killed.sum())} cells")
            cv2.imwrite(os.path.join(TUNING_DIR, "1b_contradiction.png"),
                        (killed.astype(np.uint8)) * 255)

    occ = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                           np.ones((POST_CLOSE, POST_CLOSE), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(occ)
    kept = np.zeros_like(occ)
    n_line = n_small = n_blob = 0
    line_like = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_CC_CELLS:
            n_small += 1
            continue
        comp = labels == i
        pts = np.argwhere(comp)[:, ::-1].astype(np.float32)
        (_, _), (W, H), _ = cv2.minAreaRect(pts)
        length, width = max(W, H), min(W, H)
        if length >= MIN_SEG_LEN_PX and width <= MAX_SEG_WIDTH_PX:
            kept[comp] = 255
            line_like.append(i)
            n_line += 1
        else:
            n_blob += 1
    # second pass: readmit rejected blobs that TOUCH kept walls (junctions,
    # corners, drift-thickened sections)
    kept_dil = cv2.dilate(kept, np.ones((5, 5), np.uint8))
    n_join = 0
    for i in range(1, n):
        if i in line_like:
            continue
        if stats[i, cv2.CC_STAT_AREA] < MIN_CC_CELLS:
            continue
        comp = labels == i
        if (kept_dil[comp] > 0).any():
            kept[comp] = 255
            n_join += 1
    print(f"[clean] kept {n_line} line-like + {n_join} joined; "
          f"dropped {n_small} tiny + {n_blob - n_join} blobs")
    cv2.imwrite(os.path.join(TUNING_DIR, "2_structural.png"), kept)

    if RECT_PRUNE:
        # dominant orientation from the wall points themselves
        pts = np.argwhere(kept > 0)[:, ::-1].astype(np.float32)
        if len(pts) > 50:
            angle = cv2.minAreaRect(pts)[2]
            if angle > 45:
                angle -= 90
            center = (WORK_PX / 2.0, WORK_PX / 2.0)
            R = cv2.getRotationMatrix2D(center, angle, 1.0)
            Rinv = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rot = cv2.warpAffine(kept, R, (WORK_PX, WORK_PX),
                                 flags=cv2.INTER_NEAREST)
            kh = cv2.getStructuringElement(cv2.MORPH_RECT, (RECT_LEN, 1))
            kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, RECT_LEN))
            runs = cv2.bitwise_or(
                cv2.morphologyEx(rot, cv2.MORPH_OPEN, kh),
                cv2.morphologyEx(rot, cv2.MORPH_OPEN, kv))
            pruned = cv2.warpAffine(runs, Rinv, (WORK_PX, WORK_PX),
                                    flags=cv2.INTER_NEAREST)
            pruned = cv2.bitwise_and(kept, pruned)   # prune-only guarantee
            shaved = int((kept > 0).sum() - (pruned > 0).sum())
            print(f"[rect] dominant angle {angle:.1f} deg; shaved "
                  f"{shaved} bump cells (runs < {RECT_LEN} cells)")
            cv2.imwrite(os.path.join(TUNING_DIR, "2b_rect_pruned.png"),
                        pruned)
            kept = pruned

    if not USE_LINE_FIT:
        return cv2.dilate(kept, np.ones((WALL_THICK_PX - 1,) * 2, np.uint8))

    try:
        skel = cv2.ximgproc.thinning(kept)
    except AttributeError:
        print("[clean] cv2.ximgproc missing; Hough on raw structural map")
        skel = kept

    lines = cv2.HoughLinesP(skel, 1, np.pi / 180, threshold=20,
                            minLineLength=10, maxLineGap=BRIDGE_PX)
    if lines is None:
        print("[clean] no Hough lines; submitting structural map")
        return cv2.dilate(kept, np.ones((WALL_THICK_PX - 1,) * 2, np.uint8))

    angles = np.array([np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]) % np.pi
                       for l in lines])
    dom = np.arctan2(np.sin(2 * angles).mean(),
                     np.cos(2 * angles).mean()) / 2.0
    canvas = np.zeros_like(kept)
    snapped_segments = []
    for l in lines[:, 0]:
        x1, y1, x2, y2 = map(float, l)
        ang = np.arctan2(y2 - y1, x2 - x1) % np.pi
        for t in (dom % np.pi, (dom + np.pi / 2) % np.pi):
            d = min(abs(ang - t), np.pi - abs(ang - t))
            if d < np.deg2rad(SNAP_DEG):
                cxm, cym = (x1 + x2) / 2, (y1 + y2) / 2
                half = np.hypot(x2 - x1, y2 - y1) / 2 + BRIDGE_PX / 2
                dx, dy = np.cos(t), np.sin(t)
                x1, y1 = cxm - dx * half, cym - dy * half
                x2, y2 = cxm + dx * half, cym + dy * half
                break
        cv2.line(canvas, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), 255, WALL_THICK_PX)
        snapped_segments.append((x1, y1, x2, y2))
    cv2.imwrite(os.path.join(TUNING_DIR, "3_line_fitted.png"), canvas)

    # --- TRAJECTORY-GATED GAP EXTENSION ---
    if GAP_EXTEND and traj_rc is not None and len(traj_rc) > 1:
        corridor = np.zeros_like(canvas)
        for k in range(1, len(traj_rc)):
            cv2.line(corridor, (traj_rc[k - 1][1], traj_rc[k - 1][0]),
                     (traj_rc[k][1], traj_rc[k][0]), 255,
                     2 * TRAJ_CLEAR_CELLS + 1)
        n_filled = n_door = 0
        for x1, y1, x2, y2 in snapped_segments:
            for (ex, ey, ox, oy) in ((x2, y2, x1, y1), (x1, y1, x2, y2)):
                L = np.hypot(ex - ox, ey - oy)
                if L < 1:
                    continue
                dx, dy = (ex - ox) / L, (ey - oy) / L
                hit = None
                blocked = False
                for step in range(2, MAX_EXTEND_CELLS + 1):
                    px = int(round(ex + dx * step))
                    py = int(round(ey + dy * step))
                    if not (0 <= px < WORK_PX and 0 <= py < WORK_PX):
                        break
                    if corridor[py, px] > 0:
                        blocked = True     # doorway: the drone flew here
                        break
                    # met another wall? (small perpendicular tolerance)
                    y0, y1_ = max(py - 1, 0), min(py + 2, WORK_PX)
                    x0, x1_ = max(px - 1, 0), min(px + 2, WORK_PX)
                    if canvas[y0:y1_, x0:x1_].any() and step > 3:
                        hit = step
                        break
                if hit is not None and not blocked:
                    cv2.line(canvas, (int(round(ex)), int(round(ey))),
                             (int(round(ex + dx * hit)),
                              int(round(ey + dy * hit))),
                             255, WALL_THICK_PX)
                    n_filled += 1
                elif blocked:
                    n_door += 1
        print(f"[gap] extensions filled: {n_filled}, aborted at drone "
              f"corridor (protected doorways): {n_door}")
        cv2.imwrite(os.path.join(TUNING_DIR, "3b_gap_extended.png"), canvas)
        cv2.imwrite(os.path.join(TUNING_DIR, "3c_drone_corridor.png"),
                    corridor)
    return canvas


def finalize(walls):
    canvas = np.full((OUT_PX, OUT_PX), 255, dtype=np.uint8)
    ys, xs = np.nonzero(walls)
    if ys.size == 0:
        print("[final] WARNING: blank map.")
        return canvas
    r0, r1 = ys.min(), ys.max() + 1
    c0, c1 = xs.min(), xs.max() + 1
    crop = walls[r0:r1, c0:c1]
    ch, cw = crop.shape
    print(f"[final] wall content: {cw * MAP_RES_M:.1f} x "
          f"{ch * MAP_RES_M:.1f} m")
    if ch > OUT_PX or cw > OUT_PX:
        print("[final] WARNING: content exceeds 30x30 m — check odometry "
              "scale. Cropping; fix before submit.")
        crop = crop[:min(ch, OUT_PX), :min(cw, OUT_PX)]
        ch, cw = crop.shape
    ro, co = (OUT_PX - ch) // 2, (OUT_PX - cw) // 2
    canvas[ro:ro + ch, co:co + cw][crop > 0] = 0
    return canvas


# ============================= VISUALIZATION =================================
def draw_camera_view(frame, mask):
    vis = frame.copy()
    overlay = vis.copy()
    overlay[mask == 0] = (0, 200, 0)
    overlay[mask > 0] = (255, 80, 0)
    vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0)
    h, w = vis.shape[:2]
    cv2.drawMarker(vis, (w // 2, h // 2), (0, 255, 255),
                   cv2.MARKER_CROSS, 30, 2)
    return cv2.resize(vis, (VIS_CAM_W, int(h * VIS_CAM_W / w)))


def render_map(wall, seen_free, traj_rc, drone_rc, yaw,
               frame_idx, total, follow=True):
    img = np.full((WORK_PX, WORK_PX, 3), 30, dtype=np.uint8)
    img[seen_free & ~wall] = (60, 130, 60)
    img[wall] = (0, 60, 255)               # every stamp visible immediately

    cx = cy = WORK_PX // 2
    cv2.line(img, (cx, cy), (cx + 20, cy), (0, 0, 200), 1)
    cv2.line(img, (cx, cy), (cx, cy - 20), (0, 200, 0), 1)
    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
    for k in range(1, len(traj_rc)):
        cv2.line(img, (traj_rc[k - 1][1], traj_rc[k - 1][0]),
                 (traj_rc[k][1], traj_rc[k][0]), (200, 120, 0), 1)
    if drone_rc is not None:
        r, c = drone_rc
        if 0 <= r < WORK_PX and 0 <= c < WORK_PX:
            cv2.circle(img, (c, r), 5, (0, 255, 0), -1)
            cv2.arrowedLine(img, (c, r),
                            (int(c + 18 * np.sin(yaw)),
                             int(r - 18 * np.cos(yaw))),
                            (255, 255, 255), 2, tipLength=0.35)

    if follow and drone_rc is not None:
        half = VIS_VIEW_PX // 2
        r0 = int(np.clip(drone_rc[0] - half, 0, WORK_PX - VIS_VIEW_PX))
        c0 = int(np.clip(drone_rc[1] - half, 0, WORK_PX - VIS_VIEW_PX))
        view = img[r0:r0 + VIS_VIEW_PX, c0:c0 + VIS_VIEW_PX]
        view = cv2.resize(view, None, fx=VIS_ZOOM, fy=VIS_ZOOM,
                          interpolation=cv2.INTER_NEAREST)
        mode = "FOLLOW"
    else:
        view = cv2.resize(img, (int(VIS_VIEW_PX * VIS_ZOOM),) * 2,
                          interpolation=cv2.INTER_AREA)
        mode = "OVERVIEW"

    hud = [f"{mode}  frame {frame_idx}/{total}",
           f"wall px: {int(wall.sum())}   free px: "
           f"{int(seen_free.sum())}",
           "q quit | p pause | s snapshot | f follow/overview"]
    for j, line in enumerate(hud):
        cv2.putText(view, line, (10, 22 + 20 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


# ============================= MAIN ==========================================
class WallMapper:
    """Streaming form of run_mapping: identical state machine, but fed one
    (frame, pose, wall_mask) at a time so it can run inside the odometry
    loop. feed() is called once per SAVED pose row, in order; the row
    counter drives FRAME_STRIDE exactly like the legacy CSV loop."""

    def __init__(self, total_hint=0):
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        os.makedirs(TUNING_DIR, exist_ok=True)
        self.global_wall = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.global_free = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_wall = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_free = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_rows = 0
        self.off_r, self.off_c = 0, 0
        self.n_aligned = 0
        self.traj_rc = []
        self.n_frames = 0
        self.snap = None
        self.paused, self.follow = False, True
        self.foot_lock = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.row_idx = 0
        self.total = total_hint
        self.quit = False

    def _clear_foot(self, grid_wall, grid_free, rc, half):
        r0 = max(rc[0] - half, 0); r1 = min(rc[0] + half + 1, WORK_PX)
        c0 = max(rc[1] - half, 0); c1 = min(rc[1] + half + 1, WORK_PX)
        grid_wall[r0:r1, c0:c1] = False
        grid_free[r0:r1, c0:c1] = True
        self.foot_lock[r0:r1, c0:c1] = True

    def _finish_submap(self):
        if self.sub_rows == 0:
            return
        ndr, ndc, ok = align_submap(self.global_wall, self.sub_wall,
                                    self.off_r, self.off_c)
        if ok:
            self.n_aligned += 1
            print(f"[submap] aligned: offset ({ndr},{ndc}) cells "
                  f"= ({ndc*MAP_RES_M*1000:+.0f},"
                  f"{-ndr*MAP_RES_M*1000:+.0f}) mm")
        else:
            print(f"[submap] pasted at carried offset "
                  f"({self.off_r},{self.off_c}) "
                  f"(insufficient wall content or overlap)")
        self.off_r, self.off_c = ndr, ndc
        composite(self.global_wall, self.global_free,
                  self.sub_wall, self.sub_free, self.off_r, self.off_c)
        self.global_wall[self.foot_lock] = False
        self.global_free[self.foot_lock] = True
        self.sub_wall = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_free = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_rows = 0

    def feed(self, frame, pose_row, mask):
        """One saved pose row. mask may be None on strided-out rows
        (it is not used then), matching the legacy loop which skipped
        those rows entirely."""
        i = self.row_idx
        self.row_idx += 1
        if i % FRAME_STRIDE:
            return
        if frame is None:
            return
        h, w = frame.shape[:2]
        pose = (pose_row["global_x_mm"], pose_row["global_y_mm"],
                pose_row["altitude_z_mm"], pose_row["yaw_rad"])

        if SNAP_ON and self.n_frames % SNAP_REFRESH == 0:
            self.snap = build_snap_lookup(self.sub_wall)
        if accumulate(self.sub_wall, self.sub_free, mask, pose, w, h,
                      snap=self.snap):
            self.n_frames += 1
            self.sub_rows += 1
            if self.sub_rows >= SUBMAP_ROWS:
                self._finish_submap()

        dr, dc = world_to_grid(pose_row["global_x_mm"] / 1000.0,
                               pose_row["global_y_mm"] / 1000.0)
        drone_rc = (int(dr), int(dc))
        self.traj_rc.append(drone_rc)
        try:
            FOOT = cv2.getTrackbarPos("Footprint", "Live Map (submaps)")
        except cv2.error:
            FOOT = FOOT_CELLS
        self._clear_foot(self.sub_wall, self.sub_free, drone_rc, FOOT)
        self.global_wall[self.foot_lock] = False
        self.global_free[self.foot_lock] = True

        if SHOW_LIVE and self.n_frames == 1:
            cv2.namedWindow("Live Map (submaps)")
            cv2.createTrackbar("Footprint", "Live Map (submaps)",
                               FOOT_CELLS, FOOT_MAX, lambda v: None)
        if SHOW_LIVE:
            view_wall = self.global_wall.copy()
            view_free = self.global_free.copy()
            composite(view_wall, view_free, self.sub_wall, self.sub_free,
                      self.off_r, self.off_c)
            cv2.imshow("Camera (blue=wall, green=free)",
                       draw_camera_view(frame, mask))
            cv2.imshow("Live Map (submaps)",
                       render_map(view_wall, view_free, self.traj_rc,
                                  drone_rc, pose_row["yaw_rad"], i,
                                  max(self.total, i + 1), self.follow))
            key = cv2.waitKey(0 if self.paused else 1) & 0xFF
            if key == ord('q'):
                print("[vis] quit — saving what we have.")
                self.quit = True
            elif key == ord('p'):
                self.paused = not self.paused
            elif key == ord('f'):
                self.follow = not self.follow
            elif key == ord('s'):
                cv2.imwrite(os.path.join(TUNING_DIR, "snap_map.png"),
                            render_map(self.global_wall, self.global_free,
                                       self.traj_rc, drone_rc,
                                       pose_row["yaw_rad"], i,
                                       max(self.total, i + 1), self.follow))
                cv2.imwrite(os.path.join(TUNING_DIR, "snap_cam.png"),
                            draw_camera_view(frame, mask))
                print("[vis] snapshots saved.")
        elif i % 300 == 0:
            print(f"[map] row {i}  wall px: {int(self.global_wall.sum())}")

    def finish(self):
        self._finish_submap()
        try:
            FOOT = cv2.getTrackbarPos("Footprint", "Live Map (submaps)")
        except cv2.error:
            FOOT = FOOT_CELLS
        for rc in self.traj_rc:
            self._clear_foot(self.global_wall, self.global_free, rc, FOOT)
        print(f"[foot] trajectory sweep done with half-width {FOOT} cells "
              f"({(2*FOOT+1)*MAP_RES_M:.2f} m square)")
        if SHOW_LIVE:
            cv2.destroyWindow("Live Map (submaps)") if False else None
        print(f"[map] frames accumulated: {self.n_frames}, submaps aligned: "
              f"{self.n_aligned}, raw wall cells: "
              f"{int(self.global_wall.sum())}, final carried offset: "
              f"({self.off_r},{self.off_c}) cells")
        if self.n_frames == 0:
            raise RuntimeError("No frames accumulated — check paths/CSV units.")
        np.save(WALL_NPY, self.global_wall)
        np.save(os.path.join(MAPPING_DIR, "seen_free.npy"),
                self.global_free)
        walls = clean_walls(self.global_wall, seen_free=self.global_free,
                            traj_rc=self.traj_rc)
        canvas = finalize(walls)
        cv2.imwrite(os.path.join(TUNING_DIR, "r0_footprint_estimate.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
        print("Saved footprint estimate to tuning/r0_footprint_estimate.png")


def run_mapping():
    """Legacy STAGE 2: read saved frames + CSV from disk. Only needed when
    SAVE_FRAMES was on and you want to re-map without re-flying the video."""
    import pandas as pd
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(TUNING_DIR, exist_ok=True)
    print("Loading model and poses...")
    model = load_yolo(MODEL_PATH)
    df = pd.read_csv(CSV_PATH)
    print(f"[sanity] altitude above floor: "
          f"min={df['altitude_z_mm'].min() - SEG_Z0:.0f} mm, "
          f"max={df['altitude_z_mm'].max() - SEG_Z0:.0f} mm")
    span_x = (df['global_x_mm'].max() - df['global_x_mm'].min()) / 1000.0
    span_y = (df['global_y_mm'].max() - df['global_y_mm'].min()) / 1000.0
    print(f"[sanity] trajectory span: {span_x:.1f} x {span_y:.1f} m")

    mapper = WallMapper(total_hint=len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        # mask only computed for rows the mapper will use (stride)
        mask = None
        frame = None
        if i % FRAME_STRIDE == 0:
            frame = cv2.imread(os.path.join(FRAMES_DIR, row["frame_file"]))
            if frame is not None:
                mask = wall_suspicion_mask(model, frame)
        mapper.feed(frame, row, mask)
        if mapper.quit:
            break
    if SHOW_LIVE:
        cv2.destroyAllWindows()
    mapper.finish()




# ============================= VISUALIZATION =================================




# ============================= MAIN ==========================================






# =============================================================================
# ==========  STAGE 3  VECTORIZE  (rose2_walls functions, verbatim)  ==========
# =============================================================================
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self):
        g = {}
        for i in range(len(self.p)):
            g.setdefault(self.find(i), []).append(i)
        return list(g.values())


def wrap_pi(a):
    """wrap an angle into [0, pi)"""
    return a % math.pi


def ang_diff_pi(a, b):
    """smallest difference between two undirected angles"""
    d = abs(wrap_pi(a) - wrap_pi(b)) % math.pi
    return min(d, math.pi - d)


def hough_fragments(clean, cfg):
    img = (clean * 255).astype(np.uint8)
    # NOTE: the original repo passes minLineLength / maxLineGap positionally,
    # which in the OpenCV python binding lands on the `lines` argument instead.
    # Passing them by keyword, as here, is what the paper actually describes.
    lines = cv2.HoughLinesP(img,
                            rho=1,
                            theta=np.pi / 180.0,
                            threshold=cfg["hough_threshold"],
                            minLineLength=cfg["hough_min_len"],
                            maxLineGap=cfg["hough_max_gap"])
    if lines is None:
        return np.zeros((0, 4), dtype=np.float64)
    segs = lines[:, 0, :].astype(np.float64)
    print("[rose2] hough fragments: %d" % len(segs))
    return segs


def assign_directions(segs, wall_dirs):
    """snap each fragment to the nearest dominant direction (assign_orebro_direction)"""
    ang = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]) % math.pi
    D = np.array(wall_dirs)[None, :]
    d = np.abs(ang[:, None] - D)
    d = np.minimum(d, math.pi - d)
    return np.argmin(d, axis=1), ang


def link_clusters(u1, u2, v, eps):
    """
    Single-linkage clustering of near-parallel segments, which is exactly
    DBSCAN(eps, min_samples=1) on the segment-to-segment distance matrix
    used by the repo. In the rotated frame the distance is
    hypot(lateral offset, longitudinal gap), so it vectorizes cleanly.
    """
    n = len(v)
    uf = UnionFind(n)
    if n == 0:
        return []
    order = np.argsort(v)
    # only pairs within eps laterally can ever link
    for ii in range(n):
        i = order[ii]
        for jj in range(ii + 1, n):
            j = order[jj]
            dv = abs(v[j] - v[i])
            if dv > eps:
                break
            gap = max(0.0, max(u1[i], u1[j]) - min(u2[i], u2[j]))
            if math.hypot(dv, gap) <= eps:
                uf.union(i, j)
    return uf.groups()


def build_walls(segs, labels, wall_dirs, cfg):
    walls = []
    for k, theta in enumerate(wall_dirs):
        sel = np.where(labels == k)[0]
        if len(sel) == 0:
            continue
        s = segs[sel]
        d = np.array([math.cos(theta), math.sin(theta)])
        nvec = np.array([-math.sin(theta), math.cos(theta)])

        p1 = s[:, 0:2]
        p2 = s[:, 2:4]
        ua = p1 @ d
        ub = p2 @ d
        u1 = np.minimum(ua, ub)
        u2 = np.maximum(ua, ub)
        v = 0.5 * (p1 @ nvec + p2 @ nvec)
        length = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])

        # 5a. wall clusters (single linkage / DBSCAN eps=7, min_samples=1)
        groups = link_clusters(u1, u2, v, cfg["link_eps"])

        # 5b. spatial clusters: merge wall clusters whose lateral offsets are close
        reps = []
        for g in groups:
            wsum = length[g].sum() + 1e-9
            reps.append(float((v[g] * length[g]).sum() / wsum))
        uf = UnionFind(len(groups))
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if abs(reps[i] - reps[j]) < cfg["lateral_thresh"]:
                    uf.union(i, j)

        for gg in uf.groups():
            members = np.concatenate([groups[i] for i in gg])
            w = length[members]
            v_rep = float(np.median(np.repeat(v[members],
                                              np.maximum(1, w.astype(int)))))
            a, b = float(u1[members].min()), float(u2[members].max())
            if b - a < cfg["min_wall_len"]:
                continue
            walls.append(dict(dir_idx=k, theta=theta, v=v_rep, u1=a, u2=b,
                              support=float(w.sum())))

    print("[rose2] candidate walls before weighting: %d" % len(walls))
    return walls


def wall_endpoints(wall):
    theta = wall["theta"]
    d = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-math.sin(theta), math.cos(theta)])
    p1 = d * wall["u1"] + n * wall["v"]
    p2 = d * wall["u2"] + n * wall["v"]
    return p1, p2


def coverage_weight(wall, occ_dil):
    p1, p2 = wall_endpoints(wall)
    L = np.hypot(*(p2 - p1))
    n = max(2, int(L))
    t = np.linspace(0.0, 1.0, n)
    pts = p1[None, :] * (1 - t)[:, None] + p2[None, :] * t[:, None]
    h, w = occ_dil.shape
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return float(occ_dil[ys, xs].mean())


def merge_duplicates(walls, cfg):
    uf = UnionFind(len(walls))
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            if a["dir_idx"] != b["dir_idx"]:
                continue
            if abs(a["v"] - b["v"]) > cfg["merge_dist"]:
                continue
            overlap = min(a["u2"], b["u2"]) - max(a["u1"], b["u1"])
            if overlap > 0.25 * min(a["u2"] - a["u1"], b["u2"] - b["u1"]):
                uf.union(i, j)

    merged = []
    for g in uf.groups():
        members = [walls[i] for i in g]
        wsum = sum(m["weight"] * (m["u2"] - m["u1"]) for m in members) + 1e-9
        v = sum(m["v"] * m["weight"] * (m["u2"] - m["u1"]) for m in members) / wsum
        base = max(members, key=lambda m: m["weight"] * (m["u2"] - m["u1"]))
        merged.append(dict(dir_idx=base["dir_idx"], theta=base["theta"], v=v,
                           u1=min(m["u1"] for m in members),
                           u2=max(m["u2"] for m in members),
                           support=sum(m["support"] for m in members),
                           weight=max(m["weight"] for m in members)))
    return merged


def line_intersection(a, b):
    """
    Intersection of the two infinite lines carrying walls a and b.
    A wall lies on {x : x . n = v}, so this is a 2x2 solve.
    Returns None if the walls are parallel.
    """
    na = np.array([-math.sin(a["theta"]), math.cos(a["theta"])])
    nb = np.array([-math.sin(b["theta"]), math.cos(b["theta"])])
    A = np.vstack([na, nb])
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None
    return np.linalg.solve(A, np.array([a["v"], b["v"]]))


def snap_junctions(walls, cfg, allow_extend=True):
    """
    ROSE2 terminates walls at edges, that is at intersections between
    extended lines, not at the raw pixel extent. For each wall end we look
    for the nearest intersection with a non-parallel wall that the end is
    already close to, and move the end exactly onto it. Trimming overshoot
    and closing small undershoot are the same operation with opposite sign.

    All candidate intersections are computed against the pre-snap geometry
    so that edits do not cascade around a loop of walls.
    """
    tol = cfg["snap_tol"]
    frozen = [dict(w) for w in walls]
    junctions = []
    moved = 0

    for i, w in enumerate(walls):
        d = np.array([math.cos(w["theta"]), math.sin(w["theta"])])
        best = {"u1": None, "u2": None}
        bestd = {"u1": tol + 1.0, "u2": tol + 1.0}

        for j, o in enumerate(frozen):
            if i == j or o["dir_idx"] == w["dir_idx"]:
                continue
            X = line_intersection(frozen[i], o)
            if X is None:
                continue

            # the crossing has to actually sit on the other wall, otherwise
            # it is a phantom corner between two walls that never meet
            do = np.array([math.cos(o["theta"]), math.sin(o["theta"])])
            uo = float(X @ do)
            if not (o["u1"] - tol <= uo <= o["u2"] + tol):
                continue

            ux = float(X @ d)
            for end, u_end in (("u1", frozen[i]["u1"]), ("u2", frozen[i]["u2"])):
                delta = abs(ux - u_end)
                if delta <= tol and delta < bestd[end]:
                    inward = (ux > u_end) if end == "u1" else (ux < u_end)
                    if inward or allow_extend:
                        bestd[end] = delta
                        best[end] = (ux, X)

        for end in ("u1", "u2"):
            if best[end] is not None:
                w[end] = best[end][0]
                junctions.append(best[end][1])
                moved += 1

        if w["u2"] - w["u1"] <= 0:                          # degenerate, undo
            w["u1"], w["u2"] = frozen[i]["u1"], frozen[i]["u2"]

    print("[rose2] snapped %d endpoint(s) to junctions (tol=%.1f px)" % (moved, tol))
    return walls, junctions


def load_poses(path, units="auto"):
    """
    Read a pose CSV. Columns are found by name, so global_x_mm / global_y_mm
    or x / y both work. Units are taken from the column suffix unless forced.
    """
    with open(path, "r") as f:
        rows = [r.strip().split(",") for r in f if r.strip()]
    header = [h.strip().lower() for h in rows[0]]

    def find(*cands):
        for c in cands:
            for i, h in enumerate(header):
                if h == c or h.startswith(c + "_") or h.endswith("_" + c):
                    return i
        return None

    ix = find("global_x", "x", "pos_x", "tx")
    iy = find("global_y", "y", "pos_y", "ty")
    if ix is None or iy is None:
        raise ValueError("could not find x / y columns in %s (header: %s)"
                         % (path, header))

    if units == "auto":
        tag = header[ix]
        scale = 0.001 if tag.endswith("_mm") else (0.01 if tag.endswith("_cm") else 1.0)
    else:
        scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}[units]

    pts = []
    for r in rows[1:]:
        try:
            pts.append([float(r[ix]) * scale, float(r[iy]) * scale])
        except (ValueError, IndexError):
            continue
    P = np.array(pts, dtype=np.float64)
    span = P.max(axis=0) - P.min(axis=0)
    step = np.hypot(*np.diff(P, axis=0).T)
    print("[path] %d poses, %.2f x %.2f m, %.1f m travelled (units x%.3g)" %
          (len(P), span[0], span[1], step.sum(), scale))
    return P


def poses_to_px(P, res, origin_px, y_up):
    """world metres -> raster pixels. y_up flips for the usual ROS map frame."""
    xy = P / res
    x = xy[:, 0] + origin_px[0]
    y = (origin_px[1] - xy[:, 1]) if y_up else (origin_px[1] + xy[:, 1])
    return np.column_stack([x, y])


def interior_mask(occ):
    """
    Free space enclosed by the map, used only as an alignment sanity check:
    a correctly registered trajectory sits almost entirely inside the building.
    """
    k = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k)
    ff = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8)
    outside = closed.copy()
    cv2.floodFill(outside, ff, (0, 0), 2)
    inside = ((outside != 2) & (closed == 0)).astype(np.uint8)

    if inside.sum() < 0.02 * inside.size:                    # open contour, leaked
        ys, xs = np.nonzero(occ)
        hull = cv2.convexHull(np.column_stack([xs, ys]))
        inside = np.zeros_like(occ)
        cv2.fillConvexPoly(inside, hull, 1)
        inside[occ > 0] = 0
        print("[align] map contour is open, using convex hull as interior")
    return inside


def align_score(P, inside, res, origin, y_up):
    """fraction of poses landing in enclosed free space, -1 if it falls off-map"""
    px = poses_to_px(P, res, origin, y_up)
    h, w = inside.shape
    xi = np.round(px[:, 0]).astype(np.int64)
    yi = np.round(px[:, 1]).astype(np.int64)
    if xi.min() < 0 or yi.min() < 0 or xi.max() >= w or yi.max() >= h:
        return -1.0, px
    return float(inside[yi, xi].mean()), px


def refine_origin(P, inside, res, origin, y_up, radius):
    """
    Local search around a supplied origin. Deliberately only rewards poses
    landing in free space: it must not penalise poses landing on walls,
    because those are the doorway crossings we are trying to find.
    """
    best = (align_score(P, inside, res, origin, y_up)[0], origin)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            o = (origin[0] + dx, origin[1] + dy)
            sc, _ = align_score(P, inside, res, o, y_up)
            if sc > best[0]:
                best = (sc, o)
    if best[1] != origin:
        print("[align] refined origin by (%+.0f, %+.0f) px"
              % (best[1][0] - origin[0], best[1][1] - origin[1]))
    return best[1]


def report_alignment(P, occ, res, origin, y_up):
    """
    Registration between the pose frame and the raster cannot be recovered
    from these two inputs alone: a short trajectory fits inside a large map
    in many places, and the correct fit is the one that crosses walls at the
    doors, so 'avoid the walls' picks the wrong answer. This prints what the
    supplied origin implies so it can be checked against the debug overlay.
    """
    inside = interior_mask(occ)
    score, px = align_score(P, inside, res, origin, y_up)
    if score < 0:
        print("[align] WARNING trajectory falls outside the raster with "
              "origin (%.0f, %.0f), y_up=%s" % (origin[0], origin[1], y_up))
        return px, inside

    occ_d = cv2.dilate(occ, np.ones((5, 5), np.uint8)) > 0
    xi = np.round(px[:, 0]).astype(np.int64)
    yi = np.round(px[:, 1]).astype(np.int64)
    print("[align] origin px (%.0f, %.0f) y_up=%s | in free space %.1f%% | "
          "on walls %.1f%% | path bbox x %.0f-%.0f y %.0f-%.0f"
          % (origin[0], origin[1], y_up, 100 * score,
             100 * occ_d[yi, xi].mean(),
             px[:, 0].min(), px[:, 0].max(), px[:, 1].min(), px[:, 1].max()))
    if score < 0.90:
        print("[align] WARNING only %.1f%% of poses are inside the mapped area. "
              "Check --debug overlay before trusting the carved doors." % (100 * score))
    return px, inside


def path_wall_crossings(walls, path_px):
    """
    Proper segment intersections between the flight path and each wall line.
    Returns, per wall index, the list of along-wall coordinates u where the
    drone got through. Anything it flew through is by definition an opening.
    """
    a = path_px[:-1]
    b = path_px[1:]
    seg = b - a
    hits = {}

    for i, wl in enumerate(walls):
        p1, p2 = wall_endpoints(wl)
        d = np.array([math.cos(wl["theta"]), math.sin(wl["theta"])])
        r = p2 - p1

        den = seg[:, 0] * (-r[1]) - seg[:, 1] * (-r[0])
        ok = np.abs(den) > 1e-9
        if not ok.any():
            continue
        diff = p1[None, :] - a
        t = np.full(len(a), -1.0)
        u = np.full(len(a), -1.0)
        t[ok] = (diff[ok, 0] * (-r[1]) - diff[ok, 1] * (-r[0])) / den[ok]
        u[ok] = (seg[ok, 0] * diff[ok, 1] - seg[ok, 1] * diff[ok, 0]) / den[ok]
        cross = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if not cross.any():
            continue
        X = a[cross] + seg[cross] * t[cross][:, None]
        hits[i] = sorted(float(x @ d) for x in X)
    return hits


def merge_intervals(iv):
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def subtract_intervals(span, cuts):
    """[u1,u2] minus a set of openings -> the wall pieces that survive"""
    pieces = [list(span)]
    for cs, ce in merge_intervals(cuts):
        nxt = []
        for s, e in pieces:
            if ce <= s or cs >= e:
                nxt.append([s, e])
                continue
            if cs > s:
                nxt.append([s, cs])
            if ce < e:
                nxt.append([ce, e])
        pieces = nxt
    return pieces


def carve_doors(walls, path_px, cfg, res):
    """
    Split every wall the flight path crosses, removing door_width metres
    centred on the crossing. This deliberately breaks the one-line-per-wall
    rule, because a wall with a door in it is two wall pieces.
    """
    half = 0.5 * cfg["door_width"] / res
    min_stub = cfg["min_stub_m"] / res
    hits = path_wall_crossings(walls, path_px)

    out, doors = [], []
    for i, wl in enumerate(walls):
        us = hits.get(i)
        if not us:
            out.append(wl)
            continue

        cuts = merge_intervals([(u - half, u + half) for u in us])
        for cs, ce in cuts:
            width_m = (ce - cs) * res
            if width_m > cfg["max_door_m"]:
                print("[doors] opening of %.2f m on wall %d exceeds max_door, "
                      "the path may run along this wall rather than through it"
                      % (width_m, i))
            c = 0.5 * (cs + ce)
            d = np.array([math.cos(wl["theta"]), math.sin(wl["theta"])])
            n = np.array([-math.sin(wl["theta"]), math.cos(wl["theta"])])
            X = d * c + n * wl["v"]
            doors.append(dict(wall=i, x_px=round(float(X[0]), 2),
                              y_px=round(float(X[1]), 2),
                              x_m=round(float(X[0]) * res, 3),
                              y_m=round(float(X[1]) * res, 3),
                              width_m=round(width_m, 3),
                              crossings=len(us)))

        for s, e in subtract_intervals([wl["u1"], wl["u2"]], cuts):
            if e - s < min_stub:
                continue
            piece = dict(wl)
            piece["u1"], piece["u2"] = s, e
            out.append(piece)

    print("[doors] carved %d opening(s) across %d wall(s), %d wall pieces out"
          % (len(doors), len(hits), len(out)))
    return out, doors


# The map raster is built in the pipeline's native pixel frame, which is rotated
# 90 deg from the origin-marker frame the supervisor's ground-truth map uses (same
# root cause as the victim CSV, see to_origin_marker_frame). (x,y)->(y,-x) in world
# corresponds to (col,row)->(-row,col) in pixels = a 90 deg CW image rotation. The
# canvas is square and content is centred, so rotating the whole image preserves
# centring, scale (0.05 m/px), size (600) and the pure black/white values.
# VERIFY the direction once against the supervisor's saved ground_truth_map.png; if
# the map reads mirrored/flipped, switch to cv2.ROTATE_90_COUNTERCLOCKWISE.
MAP_FRAME_ROTATE = cv2.ROTATE_90_CLOCKWISE


def render(walls, out_path, size=OUT_SIZE, thick=OUT_THICK, margin=OUT_MARGIN,
           junctions=None):
    """
    Fixed-scale render: 1 input pixel = 1 output pixel = 0.05 m, as the
    competition requires. The wall bounding box is centred on the canvas by
    translation only — NO rescaling. (The old version zoomed content to fill
    the frame, which made the submitted map read ~2x too large against the
    ground truth raster.) The finished raster is rotated into the origin-marker
    frame (MAP_FRAME_ROTATE) so it aligns with the ground-truth map for scoring.
    """
    canvas = np.full((size, size), 255, dtype=np.uint8)
    if not walls:
        cv2.imwrite(out_path, canvas)
        return canvas, 1.0, (0.0, 0.0)

    pts = np.array([p for w in walls for p in wall_endpoints(w)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    centre = 0.5 * (lo + hi)
    off_x = size / 2.0 - centre[0]
    off_y = size / 2.0 - centre[1]
    if span[0] > size or span[1] > size:
        print("[render] WARNING content %.0f x %.0f px exceeds the %d px "
              "canvas and will be clipped" % (span[0], span[1], size))

    def to_px(p):
        return (int(round(p[0] + off_x)), int(round(p[1] + off_y)))

    for w in walls:
        p1, p2 = wall_endpoints(w)
        cv2.line(canvas, to_px(p1), to_px(p2), 0, thick, cv2.LINE_8)

    if junctions is not None and thick > 2:
        for X in junctions:
            cv2.circle(canvas, to_px(X), thick // 2, 0, -1, cv2.LINE_8)

    canvas = np.where(canvas < 128, 0, 255).astype(np.uint8)
    if MAP_FRAME_ROTATE is not None:
        canvas = cv2.rotate(canvas, MAP_FRAME_ROTATE)   # -> origin-marker orientation
    cv2.imwrite(out_path, canvas)
    return canvas, 1.0, (off_x, off_y)

# =============================================================================
# =====================  STAGE 4  GROUND TRUTH COMPARISON  ====================
# =============================================================================
def load_gt(path):
    rows = list(csv.DictReader(open(path)))
    segs, meta = [], []
    for r in rows:
        if r["kind"] not in INCLUDE_KINDS:
            continue
        segs.append([float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])])
        meta.append((r["kind"], r["name"]))
    segs = np.array(segs, float)
    tot = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]).sum()
    print("[gt] %d segments (%d wall, %d window), %.1f m"
          % (len(segs), sum(1 for k, _ in meta if k == "Wall"),
             sum(1 for k, _ in meta if k == "Window"), tot))
    return segs, meta


def orient(segs, k, mirror):
    p = segs.reshape(-1, 2).copy()
    if mirror:
        p[:, 0] = -p[:, 0]
    for _ in range(k % 4):
        p = np.column_stack([-p[:, 1], p[:, 0]])
    return p.reshape(-1, 4)


def rasterise(segs, size=CANVAS, res=MAP_RES_M, thick=THICK_PX, centre=None):
    canvas = np.full((size, size), 255, np.uint8)
    if len(segs) == 0:
        return canvas
    p = segs.reshape(-1, 2)
    if centre is None:
        centre = 0.5 * (p.min(axis=0) + p.max(axis=0))
    for x1, y1, x2, y2 in segs:
        a = (int(round((x1 - centre[0]) / res + size / 2)),
             int(round(size / 2 - (y1 - centre[1]) / res)))
        b = (int(round((x2 - centre[0]) / res + size / 2)),
             int(round(size / 2 - (y2 - centre[1]) / res)))
        cv2.line(canvas, a, b, 0, thick, cv2.LINE_8)
    return np.where(canvas < 128, 0, 255).astype(np.uint8)


def best_shift(gt, est, radius=SHIFT_SEARCH_PX):
    """Overlap-maximising whole-pixel shift, one FFT correlation.
    Written numpy-version-proof: inputs forced 2D, peak located with plain
    integer divmod instead of unravel_index."""
    def _as2d(img):
        # tolerate (H,W,1), (H,W,3), (H,W,4) — some OpenCV builds return
        # a trailing singleton channel even for grayscale reads
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    gt = _as2d(gt)
    est = _as2d(est)
    G = (gt == 0).astype(np.float64)
    E = (est == 0).astype(np.float64)
    n = 2 * CANVAS
    c = np.fft.irfft2(np.fft.rfft2(G, s=(n, n)) *
                      np.conj(np.fft.rfft2(E, s=(n, n))), s=(n, n))
    c = np.roll(np.roll(c, CANVAS, axis=0), CANVAS, axis=1)
    w = c[CANVAS - radius:CANVAS + radius + 1,
          CANVAS - radius:CANVAS + radius + 1]
    flat = int(np.argmax(w))
    dy = flat // w.shape[1]
    dx = flat % w.shape[1]
    return int(dx - radius), int(dy - radius)


def shift_image(img, dx, dy, fill=255):
    out = np.full_like(img, fill)
    h, w = img.shape
    xs, xd = (0, dx) if dx >= 0 else (-dx, 0)
    ys, yd = (0, dy) if dy >= 0 else (-dy, 0)
    ww, hh = w - abs(dx), h - abs(dy)
    if ww > 0 and hh > 0:
        out[yd:yd + hh, xd:xd + ww] = img[ys:ys + hh, xs:xs + ww]
    return out


def metrics(gt, est):
    G, E = gt == 0, est == 0
    inter = float(np.logical_and(G, E).sum())
    union = float(np.logical_or(G, E).sum())
    tp = inter
    fp = float(E.sum()) - inter
    fn = float(G.sum()) - inter
    tn = float(G.size) - tp - fp - fn
    o = dict(gt_px=int(G.sum()), est_px=int(E.sum()),
             iou=inter / union if union else 1.0,
             precision=tp / (tp + fp) if tp + fp else 0.0,
             recall=tp / (tp + fn) if tp + fn else 0.0,
             pixel_acc=(tp + tn) / G.size)
    o["f1"] = (2 * o["precision"] * o["recall"] / (o["precision"] + o["recall"])
               if o["precision"] + o["recall"] else 0.0)
    dt_gt = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 3)
    dt_est = cv2.distanceTransform((~E).astype(np.uint8), cv2.DIST_L2, 3)
    o["band"] = {}
    for t in TOLERANCES_PX:
        p = float((dt_gt[E] <= t).mean()) if E.any() else 0.0
        r = float((dt_est[G] <= t).mean()) if G.any() else 0.0
        o["band"][t] = dict(precision=p, recall=r,
                            f1=2 * p * r / (p + r) if p + r else 0.0)
    o["chamfer_m"] = MAP_RES_M * 0.5 * ((float(dt_gt[E].mean()) if E.any() else 0.0) +
                                        (float(dt_est[G].mean()) if G.any() else 0.0))
    return o


def overlay(gt, est, ghosts=None):
    G, E = gt == 0, est == 0
    img = np.full((CANVAS, CANVAS, 3), 255, np.uint8)
    img[G & ~E] = (60, 60, 235)
    img[E & ~G] = (235, 120, 60)
    img[G & E] = (30, 30, 30)
    if ghosts is not None:
        img[ghosts] = (0, 165, 255)
    cv2.rectangle(img, (0, 0), (CANVAS - 1, 34), (255, 255, 255), -1)
    keys = [("missed", (60, 60, 235)), ("extra", (235, 120, 60)),
            ("agree", (30, 30, 30))]
    if ghosts is not None:
        keys.append(("ghost", (0, 165, 255)))
    for i, (t, c) in enumerate(keys):
        cv2.rectangle(img, (8 + i * 140, 10), (24 + i * 140, 24), c, -1)
        cv2.putText(img, t, (30 + i * 140, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (40, 40, 40), 1, cv2.LINE_AA)
    return img

# =============================================================================
# ======================  STAGE 3 DRIVER + PIPELINE  ==========================
# =============================================================================
def rose_intensity(raw):
    """Occupancy -> spectral wedge filter -> structure cut -> intensity cut.
    Returns the binary wall map (r3b) and the dominant wall directions."""
    b = (raw > 0).astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(b))
    amp = np.abs(F)

    h, w = b.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx))) % 180.0
    rad = np.hypot(yy - cy, xx - cx)

    sel = rad > LOW_FREQ_KEEP
    hist = np.zeros(180)
    np.add.at(hist, ang[sel].astype(int) % 180, amp[sel])
    hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 5), 0).ravel()
    peaks = []
    for a in np.argsort(hist)[::-1]:
        if all(min(abs(a - p), 180 - abs(a - p)) > 30 for p in peaks):
            peaks.append(int(a))
        if len(peaks) >= N_DIRECTIONS:
            break

    mask = rad <= LOW_FREQ_KEEP
    for p in peaks:
        d = np.minimum(np.abs(ang - p), 180 - np.abs(ang - p))
        mask |= d <= WEDGE_DEG
    resp = np.maximum(np.real(np.fft.ifft2(np.fft.ifftshift(F * mask))), 0)

    thresh = np.quantile(resp[raw > 0], 1.0 - KEEP_FRACTION)
    structure = ((raw > 0) & (resp >= thresh)).astype(np.uint8) * 255

    win = max(1, INT_WIN | 1)
    intensity = cv2.boxFilter(resp * (structure > 0), -1, (win, win),
                              normalize=False)
    s = structure > 0
    ithresh = float(np.quantile(intensity[s], INT_Q)) if s.any() else 0.0
    binary = (s & (intensity >= ithresh)).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(TUNING_DIR, "r3b_intensity_binary.png"), binary)
    print("[rose] ridges %s deg | raw %d -> structure %d -> binary %d px"
          % (peaks, int((raw > 0).sum()), int(s.sum()), int((binary > 0).sum())))

    wall_dirs = [wrap_pi(math.radians(p) + math.pi / 2) for p in peaks]
    return binary, wall_dirs


def run_vectorize():
    """STAGE 3: wall_raw.npy -> tuned vectorized walls -> OUT_PATH png."""
    raw = (np.load(WALL_NPY).astype(np.uint8)) * 255
    binary, wall_dirs = rose_intensity(raw)
    occ = (binary > 0).astype(np.uint8)

    # trajectory registration (for doorway carving)
    path_px = None
    if os.path.exists(CSV_PATH):
        P = load_poses(CSV_PATH, "auto")
        origin = POSE_ORIGIN_PX
        if ALIGN_REFINE and occ.sum():
            origin = refine_origin(P, interior_mask(occ), MAP_RES_M, origin,
                                   POSE_Y_UP, ALIGN_REFINE)
        path_px, _ = report_alignment(P, occ, MAP_RES_M, origin, POSE_Y_UP)
    else:
        print("[path] %s missing, doorways will not be carved" % CSV_PATH)

    segs = hough_fragments(occ, CFG)
    if len(segs) == 0:
        raise SystemExit("no hough fragments; the binary is empty")
    labels, _ = assign_directions(segs, wall_dirs)
    walls = build_walls(segs, labels, wall_dirs, CFG)

    k = np.ones((2 * CFG["cov_tol"] + 1,) * 2, np.uint8)
    occ_dil = cv2.dilate(occ, k) > 0
    kept = []
    for w in walls:
        w["weight"] = coverage_weight(w, occ_dil)
        if w["weight"] >= CFG["th1"]:
            kept.append(w)

    walls = merge_duplicates(kept, CFG)
    walls, junctions = snap_junctions(walls, CFG, allow_extend=True)
    walls = [w for w in walls if (w["u2"] - w["u1"]) >= CFG["min_wall_len"]]

    doors = []
    if path_px is not None and walls:
        walls, doors = carve_doors(walls, path_px, CFG, MAP_RES_M)

    if junctions:
        ends = [p for w in walls for p in wall_endpoints(w)]
        junctions = [X for X in junctions
                     if any(np.hypot(*(X - e)) < 1.5 for e in ends)]
    walls.sort(key=lambda w: -(w["u2"] - w["u1"]))

    if os.path.dirname(OUT_PATH):
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    _canvas, _scale, (off_x, off_y) = render(
        walls, OUT_PATH, OUT_SIZE, OUT_THICK, OUT_MARGIN, junctions)

    # map_estimate.png is centred on the WALL bounding box, because the
    # competition requires "the result should be centred in the 600 x 600
    # canvas" (README) and origin-centring would push a corner-anchored
    # building off the canvas and clip walls. The cost is that the origin
    # marker is no longer at a fixed pixel, so we record where marker (0,0)
    # actually landed. The ground controller reads this to turn its odometry
    # (metres, origin frame) into map pixels:
    #     col = origin_px[0] + x / res
    #     row = origin_px[1] - y / res      (y_up: north is up = smaller row)
    origin_px = [round(POSE_ORIGIN_PX[0] + off_x, 2),
                 round(POSE_ORIGIN_PX[1] + off_y, 2)]
    meta = dict(resolution_m_per_px=MAP_RES_M, size_px=OUT_SIZE,
                origin_px=origin_px, y_up=True, frame="origin_marker")
    with open(os.path.join(os.path.dirname(OUT_PATH),
                           "map_estimate_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[stage3] origin marker (0,0) sits at pixel %s of the map"
          % origin_px)

    rows = []
    for i, w in enumerate(walls):
        p1, p2 = wall_endpoints(w)
        L = float(np.hypot(*(p2 - p1)))
        rows.append(dict(id=i,
                         x1_px=round(float(p1[0]), 2), y1_px=round(float(p1[1]), 2),
                         x2_px=round(float(p2[0]), 2), y2_px=round(float(p2[1]), 2),
                         length_m=round(L * MAP_RES_M, 3),
                         angle_deg=round(math.degrees(w["theta"]), 2),
                         weight=round(w["weight"], 3)))
    with open(os.path.join(TUNING_DIR, "walls_vectorized.csv"), "w") as f:
        if rows:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")

    # --- WALL SEGMENTS IN METRES for the ground robots' mission planning ---
    # Same origin-marker frame as the victim CSV and the map's origin anchor.
    # A work-grid pixel (col, row) inverts world_to_grid back to metres:
    #     x_m = (col - POSE_ORIGIN_PX[0]) * res
    #     y_m = (POSE_ORIGIN_PX[1] - row) * res      (y_up: north is +y)
    # Walls are a known 0.2 m thick, so the robots inflate these lines by their
    # own radius plus the wall half-thickness when planning; no thickness column
    # is needed here.
    os.makedirs(os.path.dirname(WALLS_OUT_CSV), exist_ok=True)
    with open(WALLS_OUT_CSV, "w") as f:
        f.write("x1_m,y1_m,x2_m,y2_m,length_m\n")
        for r in rows:
            x1_m = (r["x1_px"] - POSE_ORIGIN_PX[0]) * MAP_RES_M
            y1_m = (POSE_ORIGIN_PX[1] - r["y1_px"]) * MAP_RES_M
            x2_m = (r["x2_px"] - POSE_ORIGIN_PX[0]) * MAP_RES_M
            y2_m = (POSE_ORIGIN_PX[1] - r["y2_px"]) * MAP_RES_M
            # Rotate into the origin-marker frame at source so the ground
            # controller reads walls in the same frame as the victim CSV.
            x1_m, y1_m = to_origin_marker_frame(x1_m, y1_m)
            x2_m, y2_m = to_origin_marker_frame(x2_m, y2_m)
            L = float(np.hypot(x2_m - x1_m, y2_m - y1_m))
            f.write("%.4f,%.4f,%.4f,%.4f,%.4f\n" % (x1_m, y1_m, x2_m, y2_m, L))
    print("[stage3] wrote %d wall segments (metres, origin frame) -> %s"
          % (len(rows), WALLS_OUT_CSV))

    if doors:
        with open(os.path.join(TUNING_DIR, "doors.csv"), "w") as f:
            f.write(",".join(doors[0].keys()) + "\n")
            for d in doors:
                f.write(",".join(str(v) for v in d.values()) + "\n")
    with open(os.path.join(TUNING_DIR, "map_estimate.json"), "w") as f:
        json.dump(dict(resolution_m_per_px=MAP_RES_M, geometry=CFG,
                       intensity=dict(wedge_deg=WEDGE_DEG,
                                      keep_fraction=KEEP_FRACTION,
                                      int_win=INT_WIN, int_q=INT_Q),
                       doors=doors, walls=rows), f, indent=2)
    print("[stage3] %s   %d wall segments, %d doorways, %.1f m of wall"
          % (OUT_PATH, len(walls), len(doors),
             sum(r["length_m"] for r in rows)))


def run_compare():
    """STAGE 4: score OUT_PATH against GT_CSV, write overlays."""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    got = load_gt(GT_CSV)
    segs_raw = got[0] if isinstance(got, tuple) else got
    gt = rasterise(orient(segs_raw, GT_ROT_K, GT_MIRROR))
    est = cv2.imread(OUT_PATH, cv2.IMREAD_GRAYSCALE)
    if est is None or int((est == 0).sum()) == 0:
        raise SystemExit("estimate at %s is missing or blank" % OUT_PATH)
    est = np.where(est < 128, 0, 255).astype(np.uint8)

    sh = best_shift(gt, est)
    dx, dy = int(sh[0]), int(sh[1])
    est_al = shift_image(est, dx, dy)
    m0, m1 = metrics(gt, est), metrics(gt, est_al)

    def line(m, tag):
        b = m["band"]
        return ("%-16s IoU %.3f  F1 %.3f  F1@2px %.3f  F1@4px %.3f  "
                "chamfer %.2f m" % (tag, m["iou"], m["f1"], b[2]["f1"],
                                    b[4]["f1"], m["chamfer_m"]))
    print("[stage4] shift %+d,%+d px" % (dx, dy))
    print("[stage4] " + line(m0, "as submitted"))
    print("[stage4] " + line(m1, "after shift"))
    for t in TOLERANCES_PX:
        b = m1["band"][t]
        print("[stage4]   within %d px (%.2f m): completeness %.3f  "
              "correctness %.3f" % (t, t * MAP_RES_M, b["recall"],
                                    b["precision"]))
    cv2.imwrite(os.path.join(COMPARE_DIR, "gt_map.png"), gt)
    cv2.imwrite(os.path.join(COMPARE_DIR, "overlay.png"), overlay(gt, est))
    cv2.imwrite(os.path.join(COMPARE_DIR, "overlay_aligned.png"),
                overlay(gt, est_al))
    print("[stage4] overlays in %s" % COMPARE_DIR)


def main():
    print("========== WORLD: %s ==========" % WORLD)
    print("  video        : %s" % os.path.relpath(VIDEO_PATH, _COMP_ROOT))
    print("  cache        : %s" % os.path.relpath(MAPPING_DIR, _COMP_ROOT))
    print("  deliverables : %s" % os.path.relpath(WORLD_LOGS_DIR, _COMP_ROOT))
    os.makedirs(WORLD_LOGS_DIR, exist_ok=True)
    os.makedirs(MAPPING_DIR, exist_ok=True)

    # Mirror this world's existing results into sim_logs/ FIRST, before any long
    # processing. If the run is forced (FORCE_ODOMETRY) or gets interrupted, the
    # simulation still points at THIS world instead of silently keeping whichever
    # world was mirrored last -- which is exactly how a large_world run ended up
    # loading medium_world's victims.
    if world_is_cached():
        activate_world()

    # Already processed this world? Then there is nothing else to do -- no need to
    # sit through the video pass again just to run the sim.
    if world_is_cached() and not (FORCE_ODOMETRY or FORCE_MAPPING):
        print("[world] '%s' already processed; skipping the pipeline." % WORLD)
        activate_world()
        print("[world] ready - open the matching Webots world and run.")
        return

    need_poses = FORCE_ODOMETRY or not os.path.exists(CSV_PATH)
    need_map = FORCE_MAPPING or not os.path.exists(WALL_NPY)

    if (need_poses or need_map) and FUSE_STAGES:
        # ---- fused stage 1+2: one video pass, one YOLO stream ----
        if not os.path.exists(VIDEO_PATH):
            raise SystemExit("fused stage 1+2 needs %s (and %s)"
                             % (VIDEO_PATH, IMU_CSV))
        print("\n========== STAGE 1+2 (fused): odometry + wall grid "
              "==========")
        run_odometry(VIDEO_PATH, IMU_CSV, mapper=WallMapper())
    else:
        # ---- legacy sequential path ----
        if need_poses:
            if not os.path.exists(VIDEO_PATH):
                raise SystemExit("stage 1 needs %s (and %s)"
                                 % (VIDEO_PATH, IMU_CSV))
            print("\n========== STAGE 1: visual odometry ==========")
            run_odometry(VIDEO_PATH, IMU_CSV)
        else:
            print("[stage1] %s exists, skipping (FORCE_ODOMETRY to redo)"
                  % CSV_PATH)
        if need_map or FORCE_ODOMETRY:
            print("\n========== STAGE 2: wall grid from frames ==========")
            run_mapping()
        else:
            print("[stage2] %s exists, skipping (FORCE_MAPPING to redo)"
                  % WALL_NPY)

    # ---- stage 3: vectorize ----
    print("\n========== STAGE 3: vectorize (tuned) ==========")
    run_vectorize()

    # ---- stage 4: compare ----
    if os.path.exists(GT_CSV):
        print("\n========== STAGE 4: ground truth comparison ==========")
        run_compare()
    else:
        print("[stage4] %s not found, skipping comparison" % GT_CSV)

    # ---- mirror this world's fresh results into sim_logs/ for the simulation ----
    activate_world()


if __name__ == "__main__":
    main()
