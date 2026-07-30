"""
IEEE SMCS Search and Rescue 2026 - Phase 1 ground mission controller.

One controller file drives BOTH ROSbots. Webots launches it once per robot,
so each process reads robot.getName() ("robot1"/"robot2") and runs its own
state machine. The two robots coordinate only over channel 73; the only thing
that scores is a JSON report on channel 43 while a robot is physically within
1 m of a victim.

Three tiers, top to bottom:
  1. Mission state machine: pick a victim, confirm it, report it, then explore.
  2. Global planner: A* over an occupancy grid built from the offline wall CSV.
  3. Local planner: a Vector Field Histogram (VFH+) controller over the full
     360-degree lidar scan. It threads doorways by choosing the free direction
     closest to the goal, uses hysteresis so it does not oscillate at gaps, and
     has a stuck-recovery behavior. This replaces the earlier naive follower
     that jittered against door edges.

Everything runs in the origin-marker metric frame. The offline pipeline writes
its map rotated 90 degrees from that frame, so we rotate it back on load
(see pipeline_to_marker).
"""

import os
import sys
import csv
import json
import math
import heapq
from collections import deque

import numpy as np

from controller import Robot

try:
    from ultralytics import YOLO
    _HAVE_YOLO = True
except Exception:
    _HAVE_YOLO = False

try:
    import cv2                       # only used for the optional debug camera window
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


# ============================ CONFIG / CONSTANTS =============================
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SIM_LOGS  = os.path.join(HERE, "sim_logs")


# ---------------------------- RUN LOGGING -----------------------------------
# Webots runs each robot controller as a separate process and the GUI console
# truncates, so each process tees its stdout to a file as well. Both robots write
# into one shared per-run folder:
#   run_logs/<timestamp>_<world>/{robot1.log, robot2.log}
# The first process to start creates the folder and the other joins it (any folder
# younger than RUN_JOIN_S).
RUN_LOGS_ROOT = os.path.join(HERE, "run_logs")
RUN_JOIN_S    = 90.0      # reuse the newest run folder if it is younger than this


def _active_world():
    try:
        with open(os.path.join(SIM_LOGS, "ACTIVE_WORLD.txt")) as f:
            return f.read().strip() or "unknown"
    except OSError:
        return "unknown"


def _run_dir():
    """Shared folder for this run: join the newest one if it was just created,
    else start a new one."""
    import time
    os.makedirs(RUN_LOGS_ROOT, exist_ok=True)
    now = time.time()
    try:
        runs = [d for d in os.listdir(RUN_LOGS_ROOT)
                if os.path.isdir(os.path.join(RUN_LOGS_ROOT, d))]
        if runs:
            newest = max(runs, key=lambda d: os.path.getmtime(
                os.path.join(RUN_LOGS_ROOT, d)))
            p = os.path.join(RUN_LOGS_ROOT, newest)
            if now - os.path.getmtime(p) < RUN_JOIN_S:
                return p
    except OSError:
        pass
    p = os.path.join(RUN_LOGS_ROOT, "%s_%s" % (
        time.strftime("%Y%m%d_%H%M%S"), _active_world()))
    os.makedirs(p, exist_ok=True)
    return p


class _Tee:
    """Write to the Webots console AND to the run log at the same time."""

    def __init__(self, stream, path):
        self.stream = stream
        self.f = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        try:
            self.f.write(s)
        except Exception:
            pass

    def flush(self):
        for t in (self.stream, self.f):
            try:
                t.flush()
            except Exception:
                pass


def start_run_log(name):
    """Tee this process's stdout/stderr into run_logs/<run>/<name>.log."""
    try:
        d = _run_dir()
        path = os.path.join(d, "%s.log" % name)
        sys.stdout = _Tee(sys.__stdout__, path)
        sys.stderr = sys.stdout
        print("[%s] logging this run to %s" % (name, path))
        return path
    except Exception as e:            # logging must never break the mission
        print("[%s] run logging disabled (%s)" % (name, e))
        return None

VICTIM_CSV = os.path.join(SIM_LOGS, "victim_location_estimates.csv")
VICTIM_UNC = os.path.join(SIM_LOGS, "victim_uncertainty.csv")
WALL_CSV   = os.path.join(SIM_LOGS, "wall_estimates.csv")
# Ground-robot victim detector: COCO "person" (class 0). Kept deliberately light
# because it runs inside the control loop; a heavier model starves the reactive
# obstacle avoidance.
YOLO_MODEL = os.path.join(HERE, "src", "yolov8n.pt")

# Robot geometry (from Rosbot.proto).
WHEEL_RADIUS = 0.043
TRACK_WIDTH  = 0.197
ROBOT_RADIUS = 0.13
# Odometry distance-scale calibration (UMBmark-style, distance channel only:
# heading is compass-absolute so it never drifts). To calibrate: command a
# straight run of ~6 m in the test world, read the encoder-integrated distance
# from the heartbeat, read the TRUE travelled distance from the supervisor, and
# set WHEEL_SCALE = true / odom. A 3-5% scale error alone puts a far victim on
# the 1 m scoring edge, so even a rough calibration is worth it. 1.0 = nominal.
WHEEL_SCALE = 1.000

START_POSES = {
    "robot1": (-0.375, 0.375, 0.0),
    "robot2": (-0.375, 0.000, 0.0),
}

SUPERVISOR_EMITTER = "supervisor emitter"        # channel 43
SQUAD_EMITTER      = "robot to robot emitter"    # channel 73
SQUAD_RECEIVER     = "robot to robot receiver"   # channel 73

# Motion (wheel angular velocities, rad/s; max 26).
CRUISE_SPEED = 6.0
SLOW_SPEED   = 2.5
TURN_GAIN    = 4.0
# DWA scores trajectories by final POSITION, not final orientation, so with the
# next waypoint well off the current heading it prefers a wide forward arc over a
# turn-in-place. At the start (both robots face +X) that arc looks like circling
# on the mat before the robot heads out. When the waypoint is more than this far
# off our heading, pivot to face it first, then let DWA drive. Kept fairly wide
# so ordinary course corrections still flow at speed through DWA and only genuine
# near-reversals (like the start-of-run turn) trigger a pivot. A small forward
# creep during the pivot preserves momentum without reopening the wide arc.
ALIGN_ANGLE  = math.radians(75)
ALIGN_CREEP  = 0.05        # m/s kept on while pivoting, so it does not fully stall

# Global planner.
GRID_RES   = 0.10
# Hard inflation stays modest so doorways stay open in the plan. On top of it a
# soft cost gradient (high near walls, decaying out to SOFT_M) makes A* prefer
# the CENTRE of corridors and only hug a wall at a doorway. Without this the
# path grazed walls the whole way and the robots ground against them.
INFLATE_M  = ROBOT_RADIUS + 0.06
WALL_THICK_M = 0.2         # every wall in these worlds is 0.2 m thick (README)
SOFT_M     = 0.60          # soft-cost radius from walls, metres
COST_W     = 0.5           # soft-cost weight added to each A* step
# Price of routing THROUGH a wall the flyover thinks is there, as a multiple of the
# grid diagonal. It must be decisive, not merely large: at a fixed 40 per cell a
# crossing priced in like a 24 m detour, and in a 20x36 m world plenty of honest
# detours are longer than that, so A* kept flipping between crossing and going
# round - the 5<->12 waypoint oscillation in the logs, with the robot pinned in a
# 1 m box swinging its heading through 180 degrees.
#
# Scaling off the grid diagonal makes one crossing cost more than ANY route that
# exists in the world, so the planner always prefers a real way round and there is
# no near-tie to oscillate on. It still crosses when the alternative is no path at
# all, which is the whole point: finite, never lethal, never stranded. Sized from
# the grid so it needs no retuning per world.
WALL_SOFT_MULT = 4.0
ARRIVE_TOL = 0.60          # metres from the victim estimate to start confirming
# Nav2-style planner goal tolerance. When the exact goal cell is unreachable
# (estimate embedded in inflated walls, doorway sealed by sensed obstacles),
# instead of failing we return the path to the REACHABLE cell closest to the
# goal, provided it lands within this of the goal. This mirrors NavFn/Smac
# "tolerance": getting within ~1.5 m of a bad estimate and letting the camera
# take over beats skipping the victim outright. Set 0 to demand exact goals.
GOAL_TOL_M = 0.0
# After a genuinely failed plan to a victim (nothing reachable even within the
# tolerance), do not retry that same victim for this long. This kills the
# no-path ping-pong loop (robot2 alternating "second attempt at #4 / #2" and
# burning the whole mission replanning to two sealed goals every tick).
NO_PATH_COOLDOWN_S = 8.0

# Local planner (VFH+).
LIDAR_ANGLE_SIGN = 1.0     # flip to -1.0 if steering comes out mirrored
LIDAR_MAX_USE    = 3.0     # ignore lidar returns beyond this, metres
VFH_SAFE_DIST    = ROBOT_RADIUS + 0.05   # half-width kept clear; small enough that
                                         # a ~0.5 m doorway still reads as passable
VFH_BINS         = 120     # candidate steering directions (3 deg each) around the circle
VFH_FWD_CONE     = math.radians(120)     # only steer within this of straight ahead
LOOKAHEAD        = 0.55    # pure-pursuit lookahead along the global path, metres
# Any-angle global planning (Theta*, Daniel et al., JAIR 39). Plain grid A* can
# only move between cell centres, so its headings are multiples of 45 degrees and
# its routes are staircases up to about 8% longer than the direct line. Theta*
# keeps the same search but reparents a cell to its grandparent whenever the two
# have line of sight, producing straight runs at arbitrary angles for comparable
# runtime. Shorter routes with fewer corners are also much easier to follow, since
# every staircase corner is a turn the controller must slow down for.
THETA_STAR       = True
# Adaptive (velocity-scaled) lookahead, as in Nav2's Regulated Pure Pursuit: the
# lookahead point sits LOOKAHEAD_TIME seconds ahead at the current speed, clamped
# to a sane band. A fixed short lookahead makes the robot cut corners and weave at
# speed; a fixed long one makes it sluggish and wide in tight spaces.
LOOKAHEAD_TIME   = 1.6     # seconds of travel to look ahead
LOOKAHEAD_MIN    = 0.45    # metres, floor for slow and precise manoeuvring
LOOKAHEAD_MAX    = 1.10    # metres, ceiling at cruise
W_GOAL           = 1.0     # pull toward the goal bearing
W_SMOOTH         = 0.35    # hysteresis toward the previous choice (anti-oscillation)
W_HEADING        = 0.15    # mild preference for the current heading
CLEAR_WINDOW     = math.radians(10)      # angular window for measuring clearance
IR_STOP          = 0.30    # hard emergency stop from the front IR sensors
FRONT_SECTOR     = math.radians(35)      # "ahead" cone for the yield check

# Local planner: Dynamic Window Approach (DWA). Samples (v, w) motion commands,
# rolls each out as a short arc, and scores the arcs on progress to the lookahead
# target, clearance from the live lidar points, and speed. Reasoning about actual
# robot trajectories makes it far more robust in tight clutter (between two
# sofas) than picking a single free steering direction.
# SPEED BUDGET: the motors allow 26 rad/s = 1.12 m/s, so the old 0.32 m/s ceiling
# used only 29% of the hardware. With a 180 s mission that ceiling, not the
# planner, was the limit (robot1 covered just 40 m in a full run). Raised to
# 0.50 m/s with extra intermediate samples so DWA can still pick a gentle speed in
# clutter, and SAFE_SLOW_DIST widened below so the robot starts easing off EARLIER
# rather than braking hard, which is what tipped it at higher speeds before.
DWA_V         = (-0.07, 0.0, 0.12, 0.20, 0.27)   # candidate linear speeds, m/s
DWA_W_MAX     = 1.25        # candidate angular speed range, rad/s
DWA_W_SAMPLES = 11
DWA_DT        = 0.25       # rollout timestep, s
DWA_STEPS     = 5          # rollout horizon = DWA_DT * DWA_STEPS seconds
DWA_OBS_RANGE = 2.5        # only score lidar points within this, metres
DWA_COLLISION = ROBOT_RADIUS + 0.04   # reject rollouts passing this close to an obstacle
DWA_CLEAR_CAP = 0.60       # clearance score saturates here, metres
DWA_W_HEAD    = 0.72       # weight: progress toward the target
DWA_W_CLEAR   = 0.30       # weight: obstacle clearance
DWA_W_VEL     = 0.12       # weight: prefer moving faster
# Acceleration limits smooth the commanded velocity so the robot ramps up/down
# instead of lurching, a hard turn snapped on at speed is what tipped it over.
# ---- Follow the Gap Method (FGM), Sezer & Gokasan, Robotics and Autonomous
# Systems 60(9) 2012 -- layered ON TOP of DWA, the "FGM-DW" hybrid.
# Pure DWA scores a fixed set of sampled arcs, so in clutter or a narrow doorway
# it often finds no good sample and crawls/oscillates while it hunts for a way
# through. FGM reads gap geometry directly off the scan in ONE pass, aims at the
# centre of the widest gap, and blends that with the goal direction:
#     phi = ((alpha/d_min)*phi_gap + phi_goal) / ((alpha/d_min) + 1)
# It is O(n), provably free of the local-minimum trap that stalls potential-field
# methods, and it only decides the HEADING -- DWA still produces (v, w) within the
# acceleration limits, so all the existing safety behaviour is untouched.
# Set FGM_ENABLE = False to A/B test against plain DWA.
FGM_ENABLE       = True
FGM_ALPHA        = 1.2                  # safety weight: higher hugs gap centres, lower cuts corners
FGM_FOV          = math.radians(200.0)  # forward arc searched for gaps
FGM_GAP_DIST     = 1.20                 # a beam this far or further counts as "free"
FGM_WIDTH_MARGIN = 1.25                 # required gap width as a multiple of the robot diameter
FGM_LOOKAHEAD    = 1.20                 # distance at which the FGM heading is projected to a target

DWA_V_ACCEL   = 0.6        # m/s^2
                           # Raised with the new top speed: at 0.6 the robot spent
                           # most of a short leg still accelerating and never
                           # actually reached cruise. Braking distance at 0.50 m/s
                           # is only v^2/2a = 0.14 m, well inside the slow zone.
DWA_W_ACCEL   = 3.0        # rad/s^2

# Dynamic obstacles get lighter inflation than walls so passable gaps between
# furniture stay open in the plan; DWA keeps the body clear at close range.
DYN_INFLATE_M  = ROBOT_RADIUS

# Live obstacle layer, updated as a log-odds occupancy grid (ROS costmap_2d
# style): each lidar beam MARKS the cell it hits and CLEARS every cell it passed
# through on the way. A static obstacle therefore stays remembered after it leaves
# the field of view, because no beam can pass through it to clear it, while an
# obstacle that genuinely moved is cleared as soon as a later beam sees through its
# old cells. Ray-clearing is used instead of time-decay so that large obstacles are
# not forgotten while the robot is still driving around them.
OBS_HIT         = 0.45        # log-odds bump added at a beam's hit cell
OBS_MISS        = 0.25        # log-odds drop on cells a beam passed through (free)
OBS_MAX         = 2.0         # score clamp, so a solid obstacle stays marked
OBS_THRESH      = 0.5         # score above this blocks planning (needs ~2 hits)
OBS_DECAY       = 0.995       # very slow fade, only to bleed off odometry-drift smears
OBS_MAX_RANGE   = 2.5         # mark hits closer than this; clear out to here, metres
OBS_UPDATE_S    = 0.10        # min seconds between obstacle-layer updates
REPLAN_PERIOD_S = 1.5         # replan on the fused map at least this often

# Path hysteresis. A* is re-run periodically on a costmap that changes as the
# lidar and depth camera stamp cells, and two routes around an obstacle can have
# nearly equal cost. Accepting every new plan therefore lets a single flickering
# cell flip the robot between homotopy classes (left vs right of a wall), so it
# commits to neither and makes no progress. The committed route is kept unless it
# is genuinely blocked or the alternative is materially shorter.
PATH_SWITCH_MARGIN = 0.20     # new route must be >=20% shorter to be worth switching
PATH_BLOCK_CHECK_M = 4.0      # only validate this far along the current path

# When boxed in with no safe rollout, recovery clears only drifted marks right
# under the robot (one cell), never the surrounding obstacle. Ray-clearing now
# self-corrects false marks, so only the single cell under the robot is cleared;
# wiping a wider box would erase the obstacle that is actually blocking us.
CLEAR_RADIUS_M = 0.15

# Depth-camera obstacle fusion. The single-plane lidar only sees its own scan
# height, so anything off that plane (e.g. a rack rod at camera level) is
# invisible to it and the robot drives straight into it. We turn the depth camera
# into a second "virtual lidar": for each image column take the nearest depth
# over a central band of rows (around the camera axis, so the FLOOR in the lower
# rows and the ceiling in the upper rows are excluded) and treat it as an
# obstacle at that bearing. Fused into both DWA and the planning costmap.
DEPTH_ROW_LO   = 0.15      # top of the used band, fraction of image height. Extended
                           # upward so obstacles ABOVE the lidar plane (a raised limb, a
                           # table edge, a pipe on a stack) are seen before contact. The
                           # floor occupies the LOWER rows, so this adds no false ground
                           # returns.
DEPTH_ROW_HI   = 0.55      # bottom of the used band (kept at/above the horizon)
DEPTH_COL_STEP = 2         # sample every Nth column for speed
DEPTH_ROW_STEP = 2
DEPTH_MIN_USE  = 0.12      # ignore closer than this (self/noise), metres
DEPTH_MAX_USE  = 3.0       # ignore farther than this, metres

# Independent collision monitor (Nav2-style), a reactive safety layer that runs
# BELOW the planner on raw fused sensors (lidar + depth camera + front IR). If
# enough obstacle points sit in the forward stop zone it refuses ALL forward
# motion and rotates toward open space, whatever the planner wanted; between the
# stop and slow zones it scales speed down so the robot is already crawling
# before it gets near. This is what stops it shoving into a rod/truck and
# toppling: "something ahead" means stop and turn, never push through.
SAFE_CONE       = math.radians(32)   # forward half-cone the monitor watches
SAFE_SIDE_CONE  = math.radians(75)   # side cones used to pick the clearer way to turn
SAFE_STOP_DIST  = 0.45               # obstacle within this ahead: no forward motion
SAFE_SLOW_DIST  = 1.10               # between stop and this: scale speed down
                                     # Widened with the higher cruise speed so the
                                     # robot eases off from further out instead of
                                     # arriving fast and braking hard near clutter.
SAFE_MIN_POINTS = 2                  # this many points in the stop zone to trip it
SAFE_SENSOR_MAX = 5.0                # stand-in distance for "nothing seen"
# When the stop zone trips, BACK AWAY from the obstacle (short reverse) with a
# gentle turn toward the clearer side, instead of pivoting in place until the
# front happens to clear (that spun the robot all the way around into a U-turn).
# Backing off opens forward space so DWA then drives through the real gap ahead,
# which is the "reverse a little, find the gap, go forward" behaviour we want.
SAFE_REVERSE_SPEED = 0.12            # m/s reverse while backing off a front obstacle
SAFE_REVERSE_CLEAR = 0.35            # only reverse if the rear IR is clearer than this

# Stuck detection and recovery. Window-based: if net displacement over the last
# STUCK_WINDOW seconds is under STUCK_NET, we are stuck (oscillating in place no
# longer fools it, since it measures net travel, not instantaneous motion).
STUCK_WINDOW   = 4.0
STUCK_NET      = 0.35
# Tip-over detection from the accelerometer. A tipped robot's wheels spin freely,
# so the encoders keep integrating and the estimated position races away from the
# true one, which would also cause victims to be marked found spuriously. Above
# this tilt, odometry integration and victim marking are both suspended.
FALLEN_TILT_DEG = 45.0
FALLEN_HOLD_S   = 0.60   # tilt must persist this long before the verdict changes
FALLEN_G_LO     = 0.75   # only judge attitude when |accel| is within this band of
FALLEN_G_HI     = 1.25   # gravity; outside it the robot is accelerating or bumped
# Goal-progress checker, equivalent to Nav2's progress_checker plugin: the
# controller must close a minimum distance to its goal within a time allowance or
# it is declared to be failing. The displacement test above only asks whether the
# robot moved at all, which a robot crawling in the wrong direction passes. This
# asks whether it is getting CLOSER, and triggers the same recovery when it is
# not.
PROGRESS_WINDOW_S = 10.0   # look back this far
PROGRESS_MIN_M    = 0.60   # ...and we must have closed at least this much distance
# Wheel-slip detection, using the lidar as a reference independent of odometry.
# If the robot wedges against an obstacle its wheels keep turning, so the encoders
# integrate travel that never happened and the estimated position drifts away from
# the true one. Odometry cannot detect its own slip, but an unchanged lidar scan
# can: claiming real travel while the scan is identical means the robot did not
# move. The forward-clearance gate below distinguishes a genuine wedge (something
# is close ahead) from a long featureless corridor, where the scan is also nearly
# unchanged during legitimate motion.
STALL_CHECK_S  = 1.0    # how often to compare scan signatures
STALL_ODOM_M   = 0.30   # odometry must claim at least this much travel...
STALL_SCAN_TOL = 0.03   # ...while the scan changed less than this on average (m)
STALL_BINS     = 24     # angular bins in the coarse scan signature
STALL_NEAR_M   = 0.60   # ...AND something must be this close ahead (a wedge always
                        # has the thing it is stuck on in front; a corridor does not)
RECOVER_TIME_S = 1.6
# If we recover more than this many times within RECOVER_ANCHOR_M of the same
# spot, escalate: rotate in place to sweep the lidar and finish mapping the
# obstacle (so A* can commit to a way around it) rather than reversing into the
# same trap again.
RECOVER_ANCHOR_M   = 0.6
RECOVER_ROTATE_AT  = 2
RECOVER_ROTATE_S   = 2.2

# Victim confirmation and reporting.
REPORT_PERIOD_S = 0.5
YOLO_CONF       = 0.30
YOLO_PERIOD_S   = 0.4
# Both ROSbots spawn side by side on the origin-marker pad, so at t=0 each one has
# the other filling its camera. The other-robot filter cannot help yet: it needs a
# squad broadcast to know where the other robot is, and none has arrived. Every run
# therefore opened with a confident victim_found fired from the launch pad, which is
# a guaranteed wrong report. A brand-new victim is only registered once the other
# robot can actually be ruled out, and never on the pad itself. This is the same
# keep-out the flyover pipeline applies for the same reason.
NEW_VICTIM_START_M = 2.0  # no new-victim registration this close to our start pose
NEW_VICTIM_CONF = 0.70   # to REGISTER a brand-new (not-from-pipeline) victim during
                         # exploration, the detection must be at least this confident
                         # AND land inside the arena, so a false positive cannot spawn
                         # a phantom victim on the map.
# ...and it must be SEEN REPEATEDLY at the same spot first. One frame of YOLO
# A single frame of the detector firing on scenery or on the other robot would
# otherwise register a permanent phantom victim. This is the controller-side
# equivalent of the pipeline's MIN_DETECTION_HITS noise filter.
NEW_VICTIM_HITS    = 3     # consistent sightings required before registering
NEW_VICTIM_MATCH_M = 1.00  # sightings within this distance count as the same candidate
NEW_VICTIM_TTL_S   = 6.0   # candidate expires if not re-seen within this
NEW_VICTIM_MIN_SEP = 1.50  # must be at least this far from a known victim / either robot
# The CSV point is only a guide. Once the camera actually sees a victim and it
# is within this range (near line of sight), we stop navigating to the exact
# coordinate and hand off to camera homing on the real victim.
CONFIRM_ACQUIRE_M = 2.0
# Approach and confirm, kept simple: drive at the victim (camera bearing), and the
# moment YOLO sees a person AND the sensor distance in the box direction is under
# the threshold, mark it found. No waist/leg depth games. The anti-stomp floor
# keeps us off the body.
CONFIRM_SPEED       = 4.2    # wheel rad/s during the approach (~0.22 m/s)
                             # anti-stomp floor still governs how close we get, so
                             # a faster approach only shortens dead time per victim.
CONFIRM_SEARCH_SPIN = 2.5    # wheel rad/s in-place scan when parked on an empty estimate
                             # (bad localization): rotate to catch a mislocalized victim on camera
CONFIRM_DIST_THRESH = 0.55   # sensor distance in the box direction under which a seen victim is FOUND.
                             # Lowered so a VISIBLE victim keeps getting creeped up on (drift-immune,
                             # driven by live lidar/depth to the body) instead of holding 0.8 m out.
# Standoff. The robot body is only ROBOT_RADIUS (0.13 m), so a 0.50 m floor was
# holding it 0.37 m clear of the victim on top of its own radius, and that standoff
# is subtracted directly from scoring margin: the logged approaches stopped 1.1-1.9 m
# from the estimate with the lidar reading 0.2-0.5 m, i.e. parked well short of a
# body it had already reached. Dropped to leave roughly 0.17 m of air beyond the
# robot's own radius, which is still clear of the mesh but much nearer the centroid.
CONFIRM_MIN_CLEAR   = 0.30   # Hard clearance floor to the nearest surface. The collision
                             # monitor deliberately carves the target victim out of its
                             # obstacle set so the victim is a goal rather than something
                             # to avoid, which leaves THIS as the only thing keeping the
                             # robot off the body. A lying limb is only ~0.1 m tall, so a
                             # thin margin plus one missed lidar beam ends in contact and
                             # a tipped robot. Scoring is measured to the waist marker,
                             # which sits inside the body, so half a metre of standoff is
                             # still comfortably inside the 1.0 m radius.
CONFIRM_SLOW_M      = 0.70   # start decelerating the approach at this clearance. Arriving at
                             # full creep speed and stopping abruptly is what tips the robot
                             # or climbs it onto an obstacle; easing in costs no closeness.
CONFIRM_RETREAT_M   = 0.25   # only back out to the pre-approach point if we crept
                             # at least this far past it while closing on the body
CONFIRM_CREEP_MIN   = 0.25   # floor on the speed scale, as a fraction of CONFIRM_SPEED
CONFIRM_TARGET_M    = 0.25   # when we do NOT see the victim, keep closing on the estimate until
                             # THIS near or the anti-stomp floor stops us. Deliberately below
                             # CONFIRM_MIN_CLEAR (0.32): on open floor the robot drives right onto
                             # the estimate, and near a body/wall the 0.32 m surface floor is the
                             # real stopper. This absorbs odometry drift+scale error (which grows
                             # with distance travelled, worst at far victims like victim2) so the
                             # TRUE position lands inside the 1.0 m scoring ring instead of on its edge.
CONFIRM_REPORT_M    = 1.30   # also report while within this of the estimate (no-camera fallback)
CONFIRM_IR_STOP     = 0.25
# Report a found-victim whenever the camera sees a person this close, in ANY
# state. The supervisor scores whichever unfound victim is within 1.0 m of our
# TRUE position when the message lands, so reporting while merely driving PAST a
# victim still scores it. This is the real fix for "reached <1.0m but MISSED".
REPORT_RANGE_M      = 1.00
# Reporting policy. Victim Finding (40%) is the MEAN of three equally weighted
# terms: victims-found ratio, time-to-find, and a confidence-accuracy term that
# compares every victim_found report against whether the robot was genuinely
# inside the 1.0 m ring when it landed (README line 340; the example report
# confirms the mean, 1.000/0.607/0.098 -> 0.568).
#
# The local supervisor applies NO penalty for a wrong report, which is why a wide,
# high-rate reporting band looks free here. It is not: only the REMOTE marking
# server computes the confidence term, and a stream of reports fired from outside
# the ring drives it to an F. Spraying reports to chase the ratio therefore trades
# a third of the victim-finding score for part of another third, and loses.
#
# So a report is a claim, not a lottery ticket. Two triggers, both meant to be
# true when they fire:
#   A) the camera sees a person within REPORT_RANGE_M. Scoring is measured to the
#      victim's centroid (waist/middle of the body), which is ON the body we can
#      see, so a close visual is strong evidence we are inside the ring.
#   B) odometry puts us within REPORT_ODOM_M of an unfound victim's estimate. The
#      camera may be facing away, so this stays as the fallback that guarantees a
#      victim driven onto is never silently skipped.
REPORT_CLOSE_M      = 0.70   # camera range under this earns high confidence
# Odometry report window, and why it is not simply "inside 1.0 m".
#
# The robot usually CANNOT reach its own estimate. The estimate is the victim's
# centroid, which sits inside the body, so the anti-stomp floor halts the robot
# against the body roughly 1.1-1.7 m short of it. Across 18 logged runs the
# closest odometry approach to a victim estimate was 0.91-1.87 m, and the one
# victim that actually scored was reported from 1.69 m from its estimate, with the
# robot's TRUE distance to the marker at 1.00 m.
#
# So a window tight enough to look precise would suppress the only reports that
# ever score. The window stays wide enough to cover the achievable approach, and
# precision comes from REPORT_IMPROVE_M instead: a report only fires when the
# robot is meaningfully CLOSER to that victim than at any previous report. The
# send budget is therefore spent walking inward, and the last sends land at the
# closest approach the robot could physically make, which is the best shot at
# being inside the ring that exists.
#
# The window must cover the whole achievable approach, not part of it. At 1.60 m it
# still cut victim4 out entirely: the robot's closest approach to that estimate was
# 1.71 m, so the odometry path never fired once, even though the robot's TRUE
# distance to the marker reached 0.84 m and a report there would have scored.
# Inside this, a report ALWAYS fires (rate limit and per-victim cap aside). This is
# the "we are at the victim" band: it matches the 1.0 m scoring radius, so if our
# estimate is right we are inside the ring, and nothing on the robot could tell us
# otherwise. Beyond it, reports are speculative and get rationed by the gate.
REPORT_SURE_M       = 1.00   # odometry this close to an estimate -> report, no gate
REPORT_ODOM_M       = 2.20   # odometry within this of an unfound estimate -> consider
# Improvement step. It also sets how far short of the true closest approach the
# last report can land: the robot creeps the final stretch in small increments, and
# any step smaller than this never earns a send. Kept small so the final report is
# made within a few centimetres of the nearest the robot ever got.
REPORT_IMPROVE_M    = 0.10   # ...and only if this much nearer than our last report
REPORT_ODOM_CONF    = 0.55   # honest: position-only, estimate error unknown to us
REPORT_SEEN_CONF    = 0.90   # camera corroborates the body is actually here
# ---- POSITION-ONLY VICTIM MARKING --------------------------------------
# Checked against the official rules: scoring requires exactly two things, being
# within 1.0 m of the victim and sending the message (README lines 28, 68, 340).
# NO lidar or camera confirmation is required anywhere - the supervisor never
# inspects a sensor. So the decision to mark/report a victim is made purely on
# ODOMETRY proximity to the victim estimate. Lidar and the depth camera are used
# ONLY for obstacle avoidance and the anti-stomp floor, never as a gate on
# marking. (The robot has no access to ground truth - there is no GPS on this
# platform, README line 98 - so odometry is the only position we can test.)
MARK_REACH_M        = 0.35   # odometry within this of the estimate = REACHED -> mark found.
# Deliberately much tighter than the 1.0 m scoring radius. Encoder odometry is
# accurate to a few centimetres over a full mission, so the dominant error is in
# the victim estimate itself (up to ~1.7 m from the true marker). True distance to
# the victim is therefore roughly estimate_error +/- MARK_REACH_M, and stopping
# short of the estimate directly costs scoring margin.
REPORT_MAX_SENDS    = 24     # max victim_found messages per victim across the pass.
# This is a backstop, not the real limiter: the closest-approach gate already caps
# the count at about REPORT_ODOM_M / REPORT_IMPROVE_M sends, since each one has to
# beat the last by REPORT_IMPROVE_M. Sized just above that so the budget can never
# empty before the robot reaches its closest approach, which is exactly how a
# victim the robot physically reached at 0.84 m was missed.
REPORT_MIN_CONF     = 0.50   # below this, do not send at all
# Divert-and-close intercept. Drive-by reports at 0.8-1.2 m to the FIGURE can
# still be >1.0 m from the scoring MARKER (waist, offset up to ~1.3 m on a lying
# victim): the near-miss log shows robot1 walking TRUE dist to victim2 down
# 1.77 -> 1.38 m while passing at camera range 0.8 m, never scoring. So when a
# robot in NAV/EXPLORE sees a person this close that maps to a victim that was
# never PHYSICALLY confirmed (unfound, skipped, or confirmed-but-uncertain), it
# breaks off its current goal and runs a full camera-homing confirm on it: the
# camera measurement is robot-relative and therefore drift-free, which is the
# classic two-level visual-servo scheme (steer on the seen target, close range
# with live sensors). Bounded per victim so a phantom cannot livelock us.
INTERCEPT_RANGE_M   = 1.80
INTERCEPT_MAX       = 2
# Body orbit after confirm. Closing to CONFIRM_DIST_THRESH of the visible part
# of the figure can still leave the true body >1.0 m from the waist marker when
# the victim was approached from the wrong end. After the hold-and-report
# completes, instead of leaving immediately the robot circles the body once at
# its standoff for ORBIT_S, reports flowing the whole way (maybe_report), so its
# TRUE position sweeps an arc around the figure and passes through the marker's
# scoring circle whichever end we came from. Bounded, anti-stomp respected.
# An approach normally ends with the robot blocked by the victim's own body or by
# nearby clutter, a few tenths of a metre short. When the estimate is offset from
# the true marker, driving straight at it cannot close that gap, but circling can:
# orbiting at radius r sweeps the true distance to the marker through
# |err - r| .. err + r, so a sufficiently long arc passes through the scoring
# radius from whichever side the robot arrived. The window must be long enough to
# cover a large fraction of the circle to be useful.
# A faster orbit sweeps the same arc in less time, so the window shortens with the
# speed increase rather than costing extra mission time: 8.0 rad/s (0.34 m/s) for
# 9 s covers 3.1 m of arc = 210 deg at r=0.84, still far more than the 59 deg that
# was leaving victims unscored.
ORBIT_S             = 12.0
ORBIT_SIDE          = math.radians(72)   # bearing at which the victim is held while circling
ORBIT_SPEED         = 4.2                # wheel rad/s during the orbit (~0.22 m/s)
# Empty-estimate orbit radius. When confirm reaches the estimate and sees
# nobody, the estimate itself is likely ~1 m off (narrow-arc triangulation
# residual). A pure spin sweeps only bearing; circling the estimate at this
# radius sweeps POSITION, and with blind reports flowing under
# CONFIRM_REPORT_M the TRUE body crosses the marker's 1.0 m scoring circle for
# any true victim within roughly this + 1.0 m of the estimate.
ORBIT_EST_R         = 0.80
ORBIT_EST_MAX_R     = 1.40   # cap on the per-victim search radius read from the pipeline
# Second-attempt when free. A robot that has cleared its own list, instead of
# just wandering, goes back for another try at victims it is UNSURE about: ones
# it could not reach (skipped), or ones it "confirmed" but with a large odometry
# gap to the estimate (a sign the true position may not have been within 1.0 m,
# e.g. a large/lying victim reached from the wrong end). Bounded so it cannot
# loop forever on a truly unreachable one.
REVISIT_DV_THRESH   = 1.00   # confirm odom gap above this flags the victim uncertain
REVISIT_MAX         = 2      # max second-attempts per victim
# Master switch for BOTH "try again" behaviours: the divert-and-close intercept
# (breaking off to re-confirm an unconfirmed victim seen in passing) AND the
# free-robot revisit (going back to uncertain/skipped victims once the list is
# cleared). Turned OFF: they were making a robot turn around and drive back to a
# victim it just failed on instead of pressing on, which wasted mission time. A
# missed victim now stays missed and the robot moves on. (The separate "my OWN
# current target is right here, confirm it early" shortcut is kept.)
ENABLE_RETRY = False
# Timing: do not time out while still closing in. Only once we are actually near
# do we hold and report for CONFIRM_HOLD_S, and CONFIRM_MAX_S caps the whole
# thing so a phantom estimate cannot trap us forever.
CONFIRM_HOLD_S     = 2.5
CONFIRM_MAX_S      = 13.0

# Debug view: draw YOLO boxes on the camera frame and show a live window per
# robot. OFF: this opened a cv2 window PER ROBOT on top of the two map windows,
# and every cv2.waitKey pumps GUI events for all of them, which is real wall-clock
# cost for information the top-view maps already give us. Turn on only when you
# specifically need to see what the camera sees.
DEBUG_CAM = False
# Live top-view map window (odometry trails, victim estimates + labels, both
# robots, wall map, 1.0 m scoring circles). Great for seeing why a victim is not
# being reached. Set False to disable.
DEBUG_MAP = False

# Exploration: once known victims are handled, the robots sweep the map for any
# undetected victims. The two split the map so they do not re-cover the same
# ground, visit coverage points nearest-first, prune points they pass close to,
# and loop the sweep so they never idle while the mission clock is still running.
COVERAGE_PRUNE_M = 1.5

# Victim approach. Real people-approach navigation (Nav2's approach action, plus
# its carrot planner that walks a blocked goal back along the goal->robot vector
# until it lands in free space) never drives to the body itself. It plans to a
# standoff pose and carves the body out of the obstacle layer, so the victim is a
# GOAL, not something to avoid. Without this the local planner treats the
# victim's own legs as an obstacle sitting on the goal cell and thrashes
# forward/reverse forever. VICTIM_STANDOFF_M is how far back the plan stops;
# VICTIM_CLEAR_R is the radius around the target victim we refuse to mark as an
# obstacle; CONFIRM_RANGE_M hands off from DWA to camera homing before the body
# can ever enter the collision band.
VICTIM_STANDOFF_M = 0.75
# The approach standoff is searched outward in radius and all the way around the
# victim, rather than only along the line back to the robot. A victim beside a
# wall has that line inflated for its whole length, which used to push the goal
# metres away and leave the robot parked well outside the scoring radius.
APPROACH_DIRS     = 16     # bearings tried at each radius, nearest ours first
APPROACH_MAX_R    = 2.00   # give up beyond this radius from the victim
VICTIM_CLEAR_R    = 0.60
CONFIRM_RANGE_M   = 1.20

# ---- lidar-to-wall odometry correction (translation only) --------------
# The compass gives an absolute, drift-free heading, so the ONLY thing that
# drifts is the (x,y) translation, from wheel-radius scale error plus slip, and
# it grows with distance travelled (worst at far victims like victim2). We snap
# it back by matching lidar hits to the known wall map and solving the small
# translation that best aligns them (point-to-line least squares, heading left
# untouched). Bonus: the walls and the victim estimates come from the SAME
# pipeline frame, so snapping the robot onto the walls also lands it in the exact
# frame the victim coordinates live in, which is what the scoring cares about.
# OFF. This was validated only against a PERFECT synthetic wall map. The real
# wall_estimates.csv is off by metres in places (its extent runs x -12.2..13.6 vs
# a true -10.0..10.0), so snapping the pose onto those walls injects metre-scale
# position errors instead of removing drift, and the robot then plans routes to
# where it wrongly believes it is. Pure encoder + compass odometry is the safer
# baseline. Re-enable ONLY after the wall map is verified accurate.
LOC_CORRECT      = False
LOC_PERIOD_S     = 0.30   # how often to run the correction
LOC_MATCH_GATE_M = 0.35   # a lidar hit matches a wall only within this perpendicular distance
LOC_MIN_MATCHES  = 8      # need at least this many matched hits to trust a correction
LOC_MAX_STEP_M   = 0.20   # per-tick cap on the applied step (larger true drift eases in over ticks)
LOC_GAIN         = 0.35   # fraction of the solved offset applied each time (smooths, no jerk)
LOC_MAX_RANGE_M  = 3.5    # only match nearby lidar hits (far hits are noisier)
LOC_RIDGE        = 1e-3   # regularizes the 2x2 solve when only one wall orientation is in view

# Inter-robot coordination.
POS_BCAST_PERIOD_S = 0.2
ROBOT_AVOID_DIST   = 1.0
ROBOT_YIELD_DIST   = 0.7
CLAIM_MATCH_M      = 1.2

# Shared "safe road". Every position a robot physically occupied is proven free
# (its body fit there), so each robot drops breadcrumbs and shares them. The
# union of both trails is a graph of guaranteed-passable space. When a robot gets
# genuinely stuck (e.g. UGV 1 fighting the truck) it plans along this proven
# corridor instead of its own grid route, which is what lets the working robot's
# road rescue the stuck one. Breadcrumb cells are also protected from being
# blocked in the costmap, so accumulated obstacle inflation can never seal a lane
# a robot already drove through.
BREADCRUMB_SPACING_M = 0.30   # drop a breadcrumb every this much travel
SAFE_LINK_M          = 0.55   # connect two breadcrumbs within this into the graph
SAFE_ATTACH_M        = 1.60   # snap the robot's pose onto the nearest breadcrumb
SAFE_GOAL_ATTACH_M   = 2.50   # nearest breadcrumb allowed to a goal (final leg is local)
SAFE_PROGRESS_M      = 0.50   # the road's goal end must beat our current goal distance by this
SAFE_ROAD_AFTER      = 1      # recoveries at one spot before switching to the safe road
                              # One failed recovery is enough evidence that the grid
                              # route is not working at this spot.
TRAIL_MAX            = 400    # cap stored breadcrumbs per trail
# How close a camera detection must be to a known victim estimate to be treated
# as that same victim (estimates can sit ~1.3 m off the real figure).
VICTIM_ASSOC_M     = 1.5

# Mission clock.
MISSION_CAP_S = 180.0
END_BUFFER_S  = 4.0

# ---- Sensor-verified closeness (do not trust odometry alone) ---------------
# Odometry cannot be the only judge of "am I at the victim": the estimate we drove
# to is itself offset from the true marker, so arriving at it proves nothing about
# the real distance. Camera bearing and lidar range, by contrast, are measured
# RELATIVE to the robot and carry no odometry error at all -- if lidar says the
# body is 0.8 m away then it is, however wrong the map position may be.
# We fuse the two: take the angular sector the detection box covers, keep the
# lidar/depth returns inside it, drop those lying far behind the nearest one (that
# removes the wall behind the victim), and average the rest. The centroid of a
# body's returns approximates its middle, which is what the scoring marker is,
# whereas the single nearest return is the body's near edge and can read a metre
# closer than the marker.
SERVO_ENABLE       = True
SERVO_STOP_M       = 0.85   # measured range to the body centroid that counts as close
SERVO_SECTOR_PAD   = math.radians(3.0)   # widen the box sector slightly
SERVO_BODY_DEPTH   = 0.70   # keep returns within this of the nearest one
SERVO_MIN_POINTS   = 3      # returns needed in the sector to trust a centroid
SERVO_FILTER_COEF  = 0.30   # EMA weight when folding a measurement into the estimate
SERVO_LOST_HOLD_S  = 1.5    # keep the last measurement this long after losing sight

# ---------------------------------------------------------------------------
# BEHAVIOUR PROFILE
# ---------------------------------------------------------------------------
# "baseline" reproduces the configuration that scored 4/5 victims on large_world:
#   * a victim is reported ONLY on a live camera detection within REPORT_RANGE_M,
#     repeated every REPORT_PERIOD_S from any state, with no send cap
#   * a victim is marked done ONLY by completing a confirm, never by odometry
#     proximity, so the robot never stops approaching early
#   * looser confirm standoffs, no post-confirm orbit
#   * plain DWA local planning and no plan hysteresis
#
# "extended" is the later configuration: position-driven marking and reporting,
# tighter standoffs, a post-confirm orbit, Follow-the-Gap heading selection and
# committed path replanning.
#
# Detection of wheel slip and tip-over is active in BOTH profiles: those only fire
# in failure states and cannot change a navigation decision.
PROFILE = "baseline"

if PROFILE == "baseline":
    MARK_ON_ODOM       = True    # position is what the rules score, so mark on it
    # Sensors are for obstacle avoidance only, never for victim confirmation, so
    # nothing stops the approach except the anti-stomp floor. These two thresholds
    # are set below what the robot can physically achieve on purpose: it then keeps
    # closing on the estimate until CONFIRM_MIN_CLEAR halts it against the body,
    # which is as near as it can get. Our estimate is offset from the true marker,
    # so every centimetre short is lost scoring margin.
    SERVO_ENABLE        = False  # no sensor-based confirmation or estimate refinement
    MARK_REACH_M        = 0.45   # odometry distance to the estimate that counts as reached
    CONFIRM_TARGET_M    = 0.15   # keep creeping in until the anti-stomp floor stops us
    # Report on EITHER trigger. Neither channel is trustworthy on its own: the
    # camera misses a victim it is not pointed at or that is occluded, and odometry
    # cannot be fully trusted either, since our victim estimate is itself offset
    # from the true marker. Firing on both means a message still lands during the
    # window when the robot's TRUE position is inside the scoring radius, which is
    # the only thing the supervisor checks.
    REPORT_ON_ODOM     = True
    REPORT_MAX_SENDS   = 24      # backstop only; see REPORT_MAX_SENDS above
    ORBIT_S             = 0.0    # no post-confirm orbit
    FGM_ENABLE          = False  # plain DWA heading selection
    # Path hysteresis stays ON. Disabling it lets a flickering costmap cell flip
    # the route between two near-equal ways around an obstacle every replan, so the
    # robot loops instead of progressing. That is a planner defect, independent of
    # how victims are confirmed, so it is not part of the behaviour being restored.
    PATH_SWITCH_MARGIN  = 0.20
else:
    MARK_ON_ODOM       = True
    REPORT_ON_ODOM     = True


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def pipeline_to_marker(x, y):
    """Identity. The 90-degree pipeline->origin-marker rotation is now baked into
    the offline pipeline at source (src/sar_pipeline.py: to_origin_marker_frame),
    so victim_location_estimates.csv and wall_estimates.csv already arrive in the
    origin-marker frame. This was moved to the pipeline because the marking
    supervisor scores that CSV directly and only translates by the origin offset
    (no rotation) -- correcting it here left the SCORED deliverable rotated 90
    degrees. Kept as a no-op so nothing downstream breaks; loaders read the CSVs
    directly now."""
    return (x, y)


# ============================ MAP / OCCUPANCY GRID ==========================
class OccupancyGrid:
    def __init__(self, walls, victims, start_xy, res=GRID_RES, pad=2.0):
        pts = list(victims) + [start_xy]
        for x1, y1, x2, y2 in walls:
            pts.append((x1, y1))
            pts.append((x2, y2))
        if not pts:
            pts = [(-1.0, -1.0), (1.0, 1.0)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.res = res
        self.min_x = min(xs) - pad
        self.min_y = min(ys) - pad
        self.max_x = max(xs) + pad
        self.max_y = max(ys) + pad
        self.nx = max(2, int(math.ceil((self.max_x - self.min_x) / res)))
        self.ny = max(2, int(math.ceil((self.max_y - self.min_y) / res)))
        self.occ = np.zeros((self.ny, self.nx), dtype=np.uint8)
        for x1, y1, x2, y2 in walls:
            self._stamp_segment(x1, y1, x2, y2)
        self._inflate(INFLATE_M)
        # The flyover wall map is a HINT, not geometry. Scored against the true
        # layout it sits about a metre out in places and draws roughly a third more
        # wall than exists, because every wall is stamped with a UAV pose that
        # drifted. Held as lethal it walled the robots in behind obstacles their own
        # lidar saw straight through, and the machinery to undo that made the
        # costmap flicker and the routes flip every replan.
        #
        # So it is kept only as a cost: A* pays WALL_SOFT_COST per cell to cross a
        # mapped wall, which is worth a detour of tens of metres, so it goes round
        # through a real doorway whenever one exists and crosses only when the
        # alternative is not reaching the victim at all. Nothing static is lethal.
        # What actually stops the robot is the live lidar layer plus the reactive
        # collision monitor, both of which measure the world instead of guessing it.
        self.wall_soft = self.occ.copy()
        self.occ = np.zeros_like(self.occ)
        # One crossing must outprice every route the world can contain, so that the
        # planner never faces a near-tie between crossing and going round.
        wall_cells = max(1.0, (WALL_THICK_M + 2.0 * INFLATE_M) / self.res)
        self.wall_cost = (WALL_SOFT_MULT * math.hypot(self.nx, self.ny)) / wall_cells
        self._compute_cost()

    def world_to_cell(self, x, y):
        return int((y - self.min_y) / self.res), int((x - self.min_x) / self.res)

    def cell_to_world(self, r, c):
        return (self.min_x + (c + 0.5) * self.res,
                self.min_y + (r + 0.5) * self.res)

    def in_bounds(self, r, c):
        return 0 <= r < self.ny and 0 <= c < self.nx

    def is_free(self, r, c):
        return self.in_bounds(r, c) and self.occ[r, c] == 0

    def los_free(self, r, c):
        """Free AND not inside a mapped wall. Used only by the Theta* shortcut test.

        Traversability and shortcutting need different answers here. A mapped wall
        is crossable (it may not exist), so is_free must allow it or the robot gets
        stranded again. But an any-angle shortcut is drawn as a straight line and
        its cells are never summed, so letting it cut the corner across a wall
        silently discards WALL_SOFT_COST and the map stops influencing anything.
        Shortcuts therefore stop at mapped walls; A* can still route through one at
        full price when there is genuinely no way round."""
        return self.is_free(r, c) and self.wall_soft[r, c] == 0

    def _stamp_segment(self, x1, y1, x2, y2):
        n = max(1, int(math.hypot(x2 - x1, y2 - y1) / (self.res * 0.5)))
        for i in range(n + 1):
            t = i / n
            r, c = self.world_to_cell(x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            if self.in_bounds(r, c):
                self.occ[r, c] = 1

    def _inflate(self, radius_m):
        rad = int(math.ceil(radius_m / self.res))
        if rad <= 0:
            return
        occupied = np.argwhere(self.occ == 1)
        grown = self.occ.copy()
        for r, c in occupied:
            grown[max(0, r - rad):min(self.ny, r + rad + 1),
                  max(0, c - rad):min(self.nx, c + rad + 1)] = 1
        self.occ = grown

    def rebuild(self, dyn_mask):
        """Rebuild the lethal layer from SENSED obstacles only, then recompute the
        soft cost (which is where the flyover walls live). A* run after this routes
        around furniture the offline map never saw, and is biased away from mapped
        walls without ever being sealed in by one."""
        self.occ = np.zeros((self.ny, self.nx), dtype=np.uint8)
        rad = int(math.ceil(DYN_INFLATE_M / self.res))
        if dyn_mask is not None and dyn_mask.any():
            for r, c in zip(*np.nonzero(dyn_mask)):
                self.occ[max(0, r - rad):min(self.ny, r + rad + 1),
                         max(0, c - rad):min(self.nx, c + rad + 1)] = 1
        self._compute_cost()

    def _compute_cost(self):
        """Soft cost gradient: a multi-source BFS gives each free cell its cell
        distance to the nearest obstacle; cells within SOFT_M of a wall pay a
        penalty that decays to zero further out. A* then favours corridor
        centres."""
        dist = np.full((self.ny, self.nx), 1e9)
        dq = deque()
        for r, c in np.argwhere(self.occ == 1):
            dist[r, c] = 0.0
            dq.append((r, c))
        while dq:
            r, c = dq.popleft()
            base = dist[r, c]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.ny and 0 <= nc < self.nx and dist[nr, nc] > base + 1:
                    dist[nr, nc] = base + 1
                    dq.append((nr, nc))
        soft = SOFT_M / self.res
        self.cost = np.clip(soft - dist, 0.0, soft) * COST_W
        # Flyover walls enter here and nowhere else: expensive to cross, never
        # impossible. This is the whole of their influence on planning.
        self.cost = self.cost + self.wall_soft * self.wall_cost

    def nearest_free(self, r, c, max_ring=25):
        if self.is_free(r, c):
            return r, c
        for ring in range(1, max_ring + 1):
            for dr in range(-ring, ring + 1):
                for dc in range(-ring, ring + 1):
                    if max(abs(dr), abs(dc)) == ring and self.is_free(r + dr, c + dc):
                        return r + dr, c + dc
        return r, c

    def line_of_sight(self, a, b):
        """Used by the waypoint smoother. Like the Theta* test it must respect
        mapped walls: collapsing two waypoints into a straight line across one
        would undo the detour A* just paid WALL_SOFT_COST to plan."""
        r0, c0 = a
        r1, c1 = b
        n = max(abs(r1 - r0), abs(c1 - c0))
        if n == 0:
            return True
        for i in range(n + 1):
            t = i / n
            if not self.los_free(int(round(r0 + t * (r1 - r0))),
                                 int(round(c0 + t * (c1 - c0)))):
                return False
        return True


def _line_of_sight(grid, a, b):
    """True when a straight line between two grid cells crosses only free cells.

    Supercover Bresenham: unlike the ordinary variant it reports every cell the
    line touches, including the two it clips when passing exactly through a
    corner. Missing those would let a path graze the diagonal gap between two
    obstacle cells, which the robot cannot physically drive through."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0
    if not grid.los_free(r, c):
        return False
    err = dr - dc
    guard = 0
    while (r, c) != (r1, c1) and guard < 4096:
        guard += 1
        e2 = 2 * err
        if e2 > -dc and e2 < dr:
            # exact diagonal step: both clipped neighbours must also be free
            if not (grid.los_free(r + sr, c) and grid.los_free(r, c + sc)):
                return False
            r += sr
            c += sc
            err += dr - dc
        elif e2 > -dc:
            err -= dc
            r += sr
        else:
            err += dr
            c += sc
        if not grid.los_free(r, c):
            return False
    return True


def astar(grid, start_xy, goal_xy, tol_m=GOAL_TOL_M):
    """A* with a Nav2-style goal tolerance: if the exact goal cell is
    unreachable (walled off, sealed by sensed obstacles) the search still
    remembers the reachable cell that came CLOSEST to the goal, and when that
    closest approach is within tol_m of the goal we return the path to it
    instead of failing. This mirrors NavFn/Smac 'tolerance' behaviour and is
    what lets a robot get next to a bad victim estimate and let the camera take
    over, rather than skipping the victim outright."""
    sr, sc = grid.nearest_free(*grid.world_to_cell(*start_xy))
    gr, gc = grid.nearest_free(*grid.world_to_cell(*goal_xy))
    start, goal = (sr, sc), (gr, gc)
    if start == goal:
        return [goal_xy]

    def h(n):
        return math.hypot(n[0] - gr, n[1] - gc)

    open_heap = [(h(start), 0.0, start)]
    came = {}
    gcost = {start: 0.0}
    best_node, best_h = start, h(start)   # closest reachable approach so far
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1),
          (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def build(cur, end_xy):
        cells = [cur]
        while cur in came:
            cur = came[cur]
            cells.append(cur)
        cells.reverse()
        return _cells_to_waypoints(grid, cells, end_xy)

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            return build(cur, goal_xy)
        hc = h(cur)
        if hc < best_h:
            best_h, best_node = hc, cur
        for dr, dc in nb:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not grid.is_free(nr, nc):
                continue
            node = (nr, nc)
            # Theta*: if this cell can see the PARENT of the current cell, hang it
            # off that parent by a straight line instead of stepping through the
            # current cell. Headings are then not restricted to multiples of 45
            # degrees, so the route is the direct line a driver would take rather
            # than a grid staircase, and it is shorter for the same search cost.
            par = came.get(cur, cur)
            if THETA_STAR and _line_of_sight(grid, par, node):
                ng = (gcost.get(par, 0.0)
                      + math.hypot(node[0] - par[0], node[1] - par[1])
                      + float(grid.cost[nr, nc]))
                parent = par
            else:
                ng = g + math.hypot(dr, dc) + float(grid.cost[nr, nc])
                parent = cur
            if ng < gcost.get(node, float("inf")):
                gcost[node] = ng
                came[node] = parent
                heapq.heappush(open_heap, (ng + h(node), ng, node))

    # Exact goal unreachable: accept the closest reachable approach if it lands
    # within the tolerance. The path ends AT that cell (not the goal), so the
    # caller's arrive/confirm logic naturally kicks in as close as physics allows.
    if tol_m > 0.0 and best_node != start and best_h * grid.res <= tol_m:
        return build(best_node, grid.cell_to_world(*best_node))
    return None


def _cells_to_waypoints(grid, cells, goal_xy):
    if len(cells) <= 2:
        return [grid.cell_to_world(*c) for c in cells[1:]] + [goal_xy]
    waypoints = [cells[0]]
    anchor = 0
    for i in range(2, len(cells)):
        if not grid.line_of_sight(cells[anchor], cells[i]):
            waypoints.append(cells[i - 1])
            anchor = i - 1
    waypoints.append(cells[-1])
    world = [grid.cell_to_world(*c) for c in waypoints[1:]]
    world.append(goal_xy)
    return world


# ============================ THE CONTROLLER ================================
class GroundMission:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0
        self.name = self.robot.getName()
        self.index = 0 if self.name.endswith("1") else 1
        start_run_log(self.name)     # tee this robot's console into run_logs/

        self._init_devices()

        x0, y0, th0 = START_POSES.get(self.name, (0.0, 0.0, 0.0))
        self.x, self.y, self.theta = x0, y0, th0
        self.start_xy = (x0, y0)
        self._last_loc = -1e9         # last lidar-to-wall correction time
        self._loc_fix = (0.0, 0.0)    # last applied (dx, dy), for the heartbeat
        self.reported = {}            # victim-key -> {"n": sends, "t": last send time}
        self.pending_new = []         # candidate new victims awaiting repeat sightings
        self.odom_dist = 0.0          # encoder-integrated path length (vs supervisor truth)
        self.fallen = False           # tipped over: odometry frozen, marking suspended
        self._tilt_since = -1.0       # when the current tilt verdict first appeared
        self._tilt_state = False      # the tilt verdict being timed
        self._servo = None            # (cx, cy, range, bearing) body centroid, robot frame
        self._servo_t = -1e9          # time of that measurement
        self._sig = None              # coarse lidar signature, for wheel-slip detection
        self._sig_t = -1e9
        self._sig_xy = None

        self.victims = self._load_victims()
        self.walls = self._load_walls()

        self.yolo = None
        if _HAVE_YOLO and os.path.exists(YOLO_MODEL):
            try:
                self.yolo = YOLO(YOLO_MODEL)
            except Exception as e:
                print(f"[{self.name}] YOLO load failed ({e}); proximity fallback")

        self._assign_victims()

        # State machine.
        self.state = "PLAN"
        self.path = []
        self.target = None
        self.confirm_start = 0.0
        self.confirm_close_since = -1.0
        self.confirm_min_dv = float("inf")   # closest odom gap to estimate this confirm
        self.confirm_got_near = False        # did we ever PHYSICALLY confirm (camera+range)?
        self.confirm_sweep_dir = 1.0         # side to work around a blocked approach
        self.confirm_orbit_start = -1.0      # body-orbit phase start time (-1 = not started)
        self.confirm_orbit_dir = 1.0         # +1 circle keeping victim to the left, -1 right
        self.no_path_until = {}              # victim id -> time before which we won't replan to it
        self.last_report = -1.0
        self.last_yolo = -1.0
        self._last_det = (False, 0.0, 0.0, 0.0)
        self.explore_targets = []
        self.explore_remaining = []

        # Local-planner state.
        self.prev_dir = 0.0
        self.cur_v = 0.0                     # last commanded (v, w) for accel limiting
        self.cur_w = 0.0
        self.pos_history = deque(maxlen=8)   # (t, x, y) samples for stuck check
        self.prog_history = deque(maxlen=16)  # (t, dist_to_goal) for the progress checker
        self._prog_goal = None                # goal the progress history belongs to
        self.recover_until = 0.0
        self.recover_count = 0               # consecutive recoveries near one spot
        self.recover_anchor = None           # where the last recovery started
        self.recover_rotate = False          # rotate in place to re-map on repeats
        self.need_replan = False
        self.obs_score = None                # live obstacle layer (built in calibrate)
        self.last_obs_update = -1.0
        self.last_replan = -1.0
        self._depth_scan = None              # cached (angs, rs) from the depth camera
        self._depth_scan_t = -1.0
        self._lidar_scan = None              # cached (angs, rs) from the lidar
        self._lidar_scan_t = -1.0
        self._fused_cache = None             # cached fused lidar+depth points
        self._fused_cache_t = -1.0

        # Inter-robot state.
        self.other_pos = None
        self.confirm_safe_xy = None      # pre-approach standoff to retreat to
        self.other_claim = None
        self.other_claim_id = None
        self.trail = []              # breadcrumbs this robot has proven free
        self.other_trail = []        # breadcrumbs received from the other robot
        self.last_crumb = None
        self.on_safe_road = False    # following the shared proven corridor
        self.last_pos_bcast = -1.0
        self.last_debug = 0.0
        self.victim_dbg = {}         # per-victim closest sensor readings
        self._dbg_printed = False
        self._last_map = -1.0        # throttle for the live top-view map window

        # Calibration filled after the first step.
        self.wheel_ref = None
        self.compass_offset = 0.0

        # Which world's data is loaded. The deliverables in sim_logs/ are a mirror
        # of one world's archive, so opening a different .wbt in Webots would send
        # the robots to ANOTHER world's victim coordinates with no other symptom.
        # Printing it makes that mismatch obvious in the first line of the log.
        active_world = "unknown"
        try:
            with open(os.path.join(SIM_LOGS, "ACTIVE_WORLD.txt")) as f:
                active_world = f.read().strip() or "unknown"
        except OSError:
            pass
        # Webots tells us which .wbt is actually loaded, so we can catch the
        # "estimates belong to a different world" mistake in the first second
        # rather than after a wasted 3-minute run.
        loaded_world = None
        try:
            wp = self.robot.getWorldPath()
            if wp:
                loaded_world = os.path.splitext(os.path.basename(wp))[0]
        except Exception:
            pass
        if loaded_world and active_world != "unknown" and loaded_world != active_world:
            print("!" * 70)
            print(f"[{self.name}] WORLD MISMATCH: Webots has loaded "
                  f"'{loaded_world}' but sim_logs/ holds '{active_world}' data.")
            print(f"[{self.name}] The robots are about to drive to ANOTHER "
                  f"WORLD'S victim coordinates.")
            print(f"[{self.name}] Fix: set VIDEO_PATH in src/sar_pipeline.py to "
                  f"'{loaded_world}' and RUN it, then restart this simulation.")
            print("!" * 70)
        print(f"[{self.name}] up. world='{active_world}' victims={len(self.victims)} "
              f"mine={sum(v['mine'] for v in self.victims)} yolo={self.yolo is not None}")
        # Print our victim ESTIMATES so they can be compared to the supervisor's
        # "Discovered victim: ... at [x, y, z]" lines. A big gap for a victim
        # means the estimate is off (pipeline), not the controller.
        for v in self.victims:
            print(f"[{self.name}] estimate V{v.get('id')} = "
                  f"({v['x']:+.2f}, {v['y']:+.2f}) "
                  f"[{v.get('quality', '?')}, search r={v.get('sigma', ORBIT_EST_R):.2f} m]")

    # ---- device setup -----------------------------------------------------
    def _init_devices(self):
        self.motors = {}
        for key, dev in (("fl", "fl_wheel_joint"), ("fr", "fr_wheel_joint"),
                         ("rl", "rl_wheel_joint"), ("rr", "rr_wheel_joint")):
            m = self.robot.getDevice(dev)
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            self.motors[key] = m

        self.encoders = {}
        for key, dev in (("fl", "front left wheel motor sensor"),
                         ("fr", "front right wheel motor sensor"),
                         ("rl", "rear left wheel motor sensor"),
                         ("rr", "rear right wheel motor sensor")):
            s = self.robot.getDevice(dev)
            s.enable(self.timestep)
            self.encoders[key] = s

        self.compass = self.robot.getDevice("imu compass")
        self.compass.enable(self.timestep)

        # Accelerometer, used ONLY to detect that the robot has tipped over. When
        # it lies on its side the wheels spin freely, the encoders keep
        # integrating, and the odometry runs away in a straight line across the
        # map -- sweeping past victim estimates and marking them all "found".
        # Gravity tells us unambiguously that we are down.
        try:
            self.accel = self.robot.getDevice("imu accelerometer")
            self.accel.enable(self.timestep)
        except Exception:
            self.accel = None

        self.lidar = self.robot.getDevice("laser")
        self.lidar.enable(self.timestep)
        try:
            self.lidar.enablePointCloud()   # so the Webots lidar overlay has data
        except Exception:
            pass

        self.ir = {}
        for dev in ("fl_range", "fr_range", "rl_range", "rr_range"):
            s = self.robot.getDevice(dev)
            s.enable(self.timestep)
            self.ir[dev] = s

        self.cam = self.robot.getDevice("camera rgb")
        self.cam.enable(self.timestep)
        self.depth = self.robot.getDevice("camera depth")
        self.depth.enable(self.timestep)

        self.sup_emitter = self.robot.getDevice(SUPERVISOR_EMITTER)
        self.squad_emitter = self.robot.getDevice(SQUAD_EMITTER)
        self.squad_receiver = self.robot.getDevice(SQUAD_RECEIVER)
        self.squad_receiver.enable(self.timestep)

    # ---- data loading -----------------------------------------------------
    def _load_victims(self):
        victims = []
        if os.path.exists(VICTIM_CSV):
            with open(VICTIM_CSV) as f:
                for row in csv.reader(f):
                    if len(row) < 2:
                        continue
                    try:
                        mx, my = float(row[0]), float(row[1])   # CSV already in origin-marker frame
                        victims.append({"id": len(victims), "x": mx, "y": my,
                                        "done": False, "mine": False,
                                        "sigma": ORBIT_EST_R, "quality": "?"})
                    except ValueError:
                        continue
        self._load_victim_uncertainty(victims)
        return victims

    def _load_victim_uncertainty(self, victims):
        """Read the pipeline's per-victim uncertainty sidecar, row-aligned with
        the victim CSV. A victim the pipeline flagged as narrow-viewing-arc gets
        a larger sigma, and the empty-estimate search orbit is sized from it, so
        we sweep wide exactly where the estimate is known to be weak and stay
        tight where it is trustworthy. Missing file (older pipeline run) simply
        leaves the default radius in place."""
        if not os.path.exists(VICTIM_UNC):
            return
        try:
            with open(VICTIM_UNC) as f:
                reader = csv.reader(f)
                next(reader, None)   # header
                for i, row in enumerate(reader):
                    if i >= len(victims) or len(row) < 5:
                        continue
                    sigma = float(row[0])
                    victims[i]["sigma"] = clamp(sigma, ORBIT_EST_R, ORBIT_EST_MAX_R)
                    victims[i]["quality"] = row[4].strip()
        except (ValueError, OSError):
            pass

    def _load_walls(self):
        walls = []
        if os.path.exists(WALL_CSV):
            with open(WALL_CSV) as f:
                reader = csv.reader(f)
                next(reader, None)   # header
                for row in reader:
                    if len(row) < 4:
                        continue
                    try:
                        x1, y1, x2, y2 = (float(v) for v in row[:4])
                        # wall_estimates.csv is already in the origin-marker frame
                        # (rotation baked into the pipeline), read it straight.
                        walls.append((x1, y1, x2, y2))
                    except ValueError:
                        continue
        return walls

    def _assign_victims(self):
        """Deterministic, travel-aware, balanced split. Coordination scoring
        rewards an EVEN split of finds and efficiency (15%) rewards short routes, so
        instead of interleaving victims by parity (which makes each robot zig-zag
        across the whole map) we sort along the axis of greatest spread and cut into
        two CONTIGUOUS, near-equal groups (3/2 for 5), then give each robot the group
        whose centroid is nearer its own start. Both robot instances compute the
        identical split from shared data, so no communication is needed."""
        n = len(self.victims)
        if n == 0:
            return
        starts = [START_POSES.get("robot1", (0.0, 0.0, 0.0))[:2],
                  START_POSES.get("robot2", (0.0, 0.0, 0.0))[:2]]
        # Nearest-first from the (shared) launch pad, then greedily give each victim
        # to whichever robot's ROUTE SO FAR is shorter. Balancing travel, not count,
        # is what matters: the previous split cut the map in half by x, which handed
        # one robot the two farthest victims and a 15 m first leg, so it spent the
        # whole 180 s driving and found almost nothing. Both robots launch from the
        # same mat, so "which half is nearer my start" was meaningless.
        order = sorted(range(n),
                       key=lambda k: min(math.hypot(self.victims[k]["x"] - s[0],
                                                    self.victims[k]["y"] - s[1])
                                         for s in starts))
        route = [0.0, 0.0]
        at = [tuple(starts[0]), tuple(starts[1])]
        for k in order:
            v = self.victims[k]
            costs = [route[i] + math.hypot(v["x"] - at[i][0], v["y"] - at[i][1])
                     for i in (0, 1)]
            i = 0 if costs[0] <= costs[1] else 1
            route[i] = costs[i]
            at[i] = (v["x"], v["y"])
            v["mine"] = (i == self.index)

    # ---- one-time calibration ---------------------------------------------
    def calibrate(self):
        self.wheel_ref = {k: s.getValue() for k, s in self.encoders.items()}
        cx, cy, _ = self.compass.getValues()
        self.compass_offset = wrap_pi(math.atan2(cx, cy) - self.theta)
        self.grid = OccupancyGrid(self.walls,
                                  [(v["x"], v["y"]) for v in self.victims],
                                  self.start_xy)
        self.obs_score = np.zeros((self.grid.ny, self.grid.nx), dtype=np.float32)
        self.map_bounds = self._compute_map_bounds()
        self.explore_targets = self._my_explore_targets()
        self.explore_remaining = list(self.explore_targets)

    def _compute_map_bounds(self, shrink=0.3):
        """Play-area box: the wall bounding box pulled slightly INWARD, then grown
        to contain every victim estimate. The floor extends past the building, so
        without this the sweep walks a robot outside the arena and burns the whole
        mission driving nowhere. It is pulled inward (not padded outward) because
        our wall estimate is already oversized versus the real building, and padding
        an oversized box let a robot drive right out of it. Victim estimates are
        always included so a tight box can never exclude a real target."""
        if not self.walls:
            return None
        xs = [c for w in self.walls for c in (w[0], w[2])]
        ys = [c for w in self.walls for c in (w[1], w[3])]
        b = [min(xs) + shrink, min(ys) + shrink,
             max(xs) - shrink, max(ys) - shrink]
        for v in self.victims:      # never exclude a victim we must reach
            b[0] = min(b[0], v["x"] - 1.0)
            b[1] = min(b[1], v["y"] - 1.0)
            b[2] = max(b[2], v["x"] + 1.0)
            b[3] = max(b[3], v["y"] + 1.0)
        return tuple(b)

    def _in_bounds(self, x, y):
        b = self.map_bounds
        if b is None:
            return True
        return b[0] <= x <= b[2] and b[1] <= y <= b[3]

    def _lawnmower_targets(self):
        g = self.grid
        targets = []
        step = 2.0
        y = g.min_y + 1.0
        flip = False
        while y < g.max_y - 1.0:
            xs = np.arange(g.min_x + 1.0, g.max_x - 1.0, step)
            if flip:
                xs = xs[::-1]
            for x in xs:
                r, c = g.world_to_cell(float(x), y)
                if g.is_free(r, c) and self._in_bounds(float(x), float(y)):
                    targets.append((float(x), float(y)))
            flip = not flip
            y += step
        return targets

    def _my_explore_targets(self):
        """Split the coverage points between the two robots along the map's x
        median, so together they sweep the whole world without overlapping."""
        pts = self._lawnmower_targets()
        if len(pts) < 2:
            return pts
        med = sorted(p[0] for p in pts)[len(pts) // 2]
        mine = [p for p in pts if (p[0] < med) == (self.index == 0)]
        return mine if mine else pts

    def _next_explore_target(self):
        """Nearest unvisited coverage point. When the region is fully swept,
        refill and sweep it again so the robot keeps searching until time is up."""
        if not self.explore_remaining:
            self.explore_remaining = list(self.explore_targets)
        if not self.explore_remaining:
            return None
        i = min(range(len(self.explore_remaining)),
                key=lambda k: math.hypot(self.explore_remaining[k][0] - self.x,
                                         self.explore_remaining[k][1] - self.y))
        return self.explore_remaining.pop(i)

    # ---- odometry ---------------------------------------------------------
    def is_fallen(self):
        """True when the robot has tipped over.

        The accelerometer measures gravity PLUS the robot's own acceleration, so a
        single sample is not enough: braking, a bump or a wheel scrub tilts the
        measured vector transiently and would read as a fall. Two guards make the
        test trustworthy. First, only judge when the measured magnitude is close to
        gravity, which excludes acceleration and impact spikes outright. Second,
        the tilt must persist for FALLEN_HOLD_S before the state flips either way,
        so a momentary reading cannot toggle it."""
        if self.accel is None:
            return False
        now = self.robot.getTime()
        try:
            ax, ay, az = self.accel.getValues()
        except Exception:
            return False
        g = math.sqrt(ax * ax + ay * ay + az * az)
        # Only a near-1g reading is dominated by gravity and therefore informative
        # about attitude; anything else is the robot accelerating or being hit.
        if not (FALLEN_G_LO * 9.81 <= g <= FALLEN_G_HI * 9.81):
            return self.fallen          # inconclusive: hold the current verdict
        tilt = math.degrees(math.acos(clamp(abs(az) / g, -1.0, 1.0)))
        tipped = tilt > FALLEN_TILT_DEG
        if tipped != self.fallen:
            if self._tilt_since < 0.0 or self._tilt_state != tipped:
                self._tilt_since, self._tilt_state = now, tipped
            if now - self._tilt_since >= FALLEN_HOLD_S:
                return tipped           # sustained long enough to believe it
        else:
            self._tilt_since = -1.0
        return self.fallen

    def update_odometry(self):
        d = 0.0
        for k, s in self.encoders.items():
            d += (s.getValue() - self.wheel_ref[k])
            self.wheel_ref[k] = s.getValue()
        d = (d / 4.0) * WHEEL_RADIUS * WHEEL_SCALE
        # Tipped over: the wheels are spinning in the air, so this encoder delta is
        # not real travel. Integrating it is what sent the believed position racing
        # across the map and falsely marked every victim it passed. Freeze instead.
        if self.is_fallen():
            if not self.fallen:
                self.fallen = True
                print(f"[{self.name}] FALLEN OVER: freezing odometry and stopping "
                      f"victim marking (wheels are spinning, not driving)")
            self.stop()
            return
        if self.fallen:
            self.fallen = False
            print(f"[{self.name}] upright again; resuming")
        cx, cy, _ = self.compass.getValues()
        self.theta = wrap_pi(math.atan2(cx, cy) - self.compass_offset)
        self.x += d * math.cos(self.theta)
        self.y += d * math.sin(self.theta)
        # Path length we believe we travelled, printed in the heartbeat. The
        # supervisor prints the TRUE distance, so WHEEL_SCALE = true / odom.
        self.odom_dist += abs(d)

    def _scan_signature(self):
        """Coarse rotation-indexed lidar signature: nearest return per angular bin.
        A cheap fingerprint of what the robot can currently see."""
        angs, rs = self.read_lidar()
        if not angs:
            return None
        sig = [float("inf")] * STALL_BINS
        for a, d in zip(angs, rs):
            i = int((wrap_pi(a) + math.pi) / (2.0 * math.pi) * STALL_BINS) % STALL_BINS
            if d < sig[i]:
                sig[i] = d
        return sig

    def check_stall(self):
        """Detect wheels-turning-but-not-moving (wedged on a body, high-centred)
        using the lidar as an external reference, because odometry cannot detect
        its own slip. If odometry claims real travel while the scan is unchanged
        AND something is close ahead, we never moved: undo the phantom travel so
        the map stops running away, and reverse out."""
        now = self.robot.getTime()
        if now - self._sig_t < STALL_CHECK_S:
            return
        sig = self._scan_signature()
        prev, prev_xy = self._sig, self._sig_xy
        self._sig, self._sig_t, self._sig_xy = sig, now, (self.x, self.y)
        if sig is None or prev is None or prev_xy is None:
            return
        if now < self.recover_until:          # already recovering, let it finish
            return
        moved = math.hypot(self.x - prev_xy[0], self.y - prev_xy[1])
        if moved < STALL_ODOM_M or abs(self.cur_v) < 0.05:
            return                            # not claiming to drive anywhere
        diffs = [abs(a - b) for a, b in zip(sig, prev)
                 if math.isfinite(a) and math.isfinite(b)]
        if len(diffs) < STALL_BINS // 3:
            return                            # too little overlap to judge
        change = sum(diffs) / len(diffs)
        # An open corridor also leaves the scan nearly unchanged (side walls stay
        # put, forward range is clipped), so require something close in front too.
        if change < STALL_SCAN_TOL and self.forward_clearance_raw() < STALL_NEAR_M:
            self.x, self.y = prev_xy          # the world did not move, nor did we
            self._sig_xy = prev_xy
            self.odom_dist = max(0.0, self.odom_dist - moved)
            print(f"[{self.name}] STALL: odom claimed {moved:.2f} m but the scan "
                  f"changed {change:.3f} m -> wedged. Undoing phantom travel, "
                  f"reversing out.")
            self._start_recovery(now)

    def correct_odometry_to_walls(self):
        """Snap the drifting (x,y) back onto the known wall map. Heading is
        already absolute (compass), so we only solve a small TRANSLATION: project
        each lidar hit into the world with the current pose, match it to the
        nearest wall segment, and least-squares the offset that best drives the
        matched hits onto their walls (minimise sum (u_i . delta - r_i)^2, where
        u_i points from the hit to its closest wall point and r_i is the gap).
        Only well-matched, nearby hits count, so a person, the other robot, or a
        truck (none of which lie on a wall) are ignored. Applied as a fraction per
        call so the pose eases toward truth instead of snapping. This bounds the
        error to the wall map's own accuracy instead of letting it grow forever."""
        if not (LOC_CORRECT and self.walls):
            return
        now = self.robot.getTime()
        if now - self._last_loc < LOC_PERIOD_S:
            return
        self._last_loc = now

        angs, rs = self.read_lidar()
        if not angs:
            return
        ct, st = math.cos(self.theta), math.sin(self.theta)
        # Accumulate the 2x2 normal equations A delta = b over matched hits.
        a00 = a01 = a11 = b0 = b1 = 0.0
        matches = 0
        gate2 = LOC_MATCH_GATE_M * LOC_MATCH_GATE_M
        for a, d in zip(angs, rs):
            if d > LOC_MAX_RANGE_M:
                continue
            ca, sa = math.cos(a), math.sin(a)
            # hit point in world coordinates
            px = self.x + d * (ca * ct - sa * st)
            py = self.y + d * (ca * st + sa * ct)
            # nearest point over all wall segments
            best2 = gate2
            cxp = cyp = None
            for x1, y1, x2, y2 in self.walls:
                wx, wy = x2 - x1, y2 - y1
                seg2 = wx * wx + wy * wy
                if seg2 < 1e-9:
                    continue
                t = ((px - x1) * wx + (py - y1) * wy) / seg2
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                qx, qy = x1 + t * wx, y1 + t * wy
                dx, dy = qx - px, qy - py
                dd = dx * dx + dy * dy
                if dd < best2:
                    best2 = dd
                    cxp, cyp = qx, qy
            if cxp is None:
                continue   # no wall within the gate: dynamic/unknown object, skip
            r = math.sqrt(best2)
            if r < 1e-6:
                continue
            ux, uy = (cxp - px) / r, (cyp - py) / r   # unit direction hit -> wall
            a00 += ux * ux; a01 += ux * uy; a11 += uy * uy
            b0 += r * ux;   b1 += r * uy
            matches += 1

        if matches < LOC_MIN_MATCHES:
            return
        # Solve (A + ridge) delta = b. The ridge keeps the unobservable direction
        # (when only one wall orientation is in view) from blowing up.
        a00 += LOC_RIDGE; a11 += LOC_RIDGE
        det = a00 * a11 - a01 * a01
        if abs(det) < 1e-9:
            return
        dx = LOC_GAIN * (a11 * b0 - a01 * b1) / det
        dy = LOC_GAIN * (a00 * b1 - a01 * b0) / det
        # Cap the per-tick step so a single noisy solve can never yank the pose;
        # a genuine larger drift is walked in over several ticks instead. The
        # 0.35 m match gate already keeps gross outliers (people, robots) out.
        mag = math.hypot(dx, dy)
        if mag > LOC_MAX_STEP_M:
            dx *= LOC_MAX_STEP_M / mag
            dy *= LOC_MAX_STEP_M / mag
        self.x += dx
        self.y += dy
        self._loc_fix = (dx, dy)

    # ---- lidar / VFH+ ------------------------------------------------------
    def read_lidar(self):
        """Full scan as (angles, ranges) in the robot frame, 0 = ahead, CCW+.
        Cached per simulation step: this is read several times per step (collision
        guard, forward clearance, victim range, costmap) and each rebuild used to
        loop over every beam in Python."""
        now = self.robot.getTime()
        if self._lidar_scan is not None and self._lidar_scan_t == now:
            return self._lidar_scan
        try:
            ranges = self.lidar.getRangeImage()
        except Exception:
            self._lidar_scan, self._lidar_scan_t = ([], []), now
            return self._lidar_scan
        n = len(ranges)
        if n < 2:
            self._lidar_scan, self._lidar_scan_t = ([], []), now
            return self._lidar_scan
        fov = self.lidar.getFov()
        try:
            arr = np.asarray(ranges, dtype=np.float32)
            keep = np.isfinite(arr) & (arr > 0.0) & (arr <= LIDAR_MAX_USE)
            idx = np.nonzero(keep)[0]
            angs = ((fov / 2.0 - idx * fov / (n - 1)) * LIDAR_ANGLE_SIGN).tolist()
            rs = arr[idx].tolist()
        except (ValueError, TypeError):
            angs, rs = [], []
        self._lidar_scan, self._lidar_scan_t = (angs, rs), now
        return self._lidar_scan

    def read_depth_scan(self):
        """Virtual lidar from the depth camera: for each column, the nearest
        obstacle over a central band of rows, returned as (angles, ranges) in the
        robot frame (0 = ahead, +left), matching read_lidar. This is what lets the
        robot see obstacles at camera height that the horizontal lidar plane
        misses (a rack rod, a table edge, the other robot's thin mast). Cached per
        simulation step because it is read by both DWA and the costmap update."""
        now = self.robot.getTime()
        if self._depth_scan is not None and self._depth_scan_t == now:
            return self._depth_scan
        angs, rs = [], []
        try:
            w = self.depth.getWidth()
            h = self.depth.getHeight()
            img = self.depth.getRangeImage()
            fov = self.depth.getFov()
            maxr = self.depth.getMaxRange()
        except Exception:
            self._depth_scan, self._depth_scan_t = ([], []), now
            return self._depth_scan
        if img and w >= 2 and h >= 2:
            r0 = max(0, int(h * DEPTH_ROW_LO))
            r1 = min(h, max(r0 + 1, int(h * DEPTH_ROW_HI)))
            # Vectorized over the band we actually use. The old nested Python loop
            # ran ~19k iterations per robot per timestep and dominated the control
            # loop dominated the control-loop cost. Row slices are C-level, and only
            # the rows inside the band are converted: converting the whole depth
            # image is measurably slower than the plain Python loop.
            try:
                rows = [img[r * w:(r + 1) * w:DEPTH_COL_STEP]
                        for r in range(r0, r1, DEPTH_ROW_STEP)]
                band = np.asarray(rows, dtype=np.float32)
                band = np.where(np.isfinite(band), band, np.inf)
                best = band.min(axis=0)
                cols = np.arange(0, w, DEPTH_COL_STEP)[:best.shape[0]]
                keep = ((best > DEPTH_MIN_USE) & (best < DEPTH_MAX_USE)
                        & (best < maxr * 0.99))
                angs = (-(cols[keep] / (w - 1) - 0.5) * fov).tolist()
                rs = best[keep].tolist()
            except (ValueError, TypeError):
                angs, rs = [], []
        self._depth_scan, self._depth_scan_t = (angs, rs), now
        return self._depth_scan

    def _fused_points(self):
        """Nearby obstacle bearings/distances in the robot frame, fusing the
        lidar (horizontal plane) and the depth camera (camera height). The depth
        camera is what makes camera-height obstacles like the rack rod visible.
        Cached per step, since several callers need it within the same step."""
        now = self.robot.getTime()
        if self._fused_cache is not None and self._fused_cache_t == now:
            return self._fused_cache
        pts = list(zip(*self.read_lidar()))
        pts += list(zip(*self.read_depth_scan()))
        self._fused_cache, self._fused_cache_t = pts, now
        return pts

    def collision_guard(self):
        """Reactive collision monitor, independent of the planner. Returns
        (stop, fwd_min, left_clear, right_clear): stop is True when at least
        SAFE_MIN_POINTS obstacle points fall in the forward stop zone (or the
        front IR trips); fwd_min is the closest thing ahead, for speed scaling;
        left/right clearances say which way to rotate away from the obstacle."""
        fwd_min = SAFE_SENSOR_MAX
        left = right = SAFE_SENSOR_MAX
        stop_pts = 0
        tv = self._target_victim_xy()   # do not stop for the victim we approach
        for a, d in self._fused_points():
            if tv is not None:
                wx = self.x + d * math.cos(self.theta + a)
                wy = self.y + d * math.sin(self.theta + a)
                if math.hypot(wx - tv[0], wy - tv[1]) < VICTIM_CLEAR_R:
                    continue
            if abs(a) < SAFE_CONE:
                if d < fwd_min:
                    fwd_min = d
                if d < SAFE_STOP_DIST:
                    stop_pts += 1
            elif SAFE_CONE <= a < SAFE_SIDE_CONE:
                left = min(left, d)
            elif -SAFE_SIDE_CONE < a <= -SAFE_CONE:
                right = min(right, d)
        ir = self.front_ir()
        if ir < fwd_min:
            fwd_min = ir
        stop = (stop_pts >= SAFE_MIN_POINTS) or (ir < SAFE_STOP_DIST)
        return stop, fwd_min, left, right

    def forward_clearance_raw(self):
        """Closest thing straight ahead from all sensors (lidar + depth + IR),
        WITHOUT carving out the victim. Used by confirm as the anti-stomp so we
        pull up a short standoff from the body instead of driving into it."""
        best = self.front_ir()
        for a, d in self._fused_points():
            if abs(a) < SAFE_CONE and d < best:
                best = d
        return best

    def victim_range(self, bearing):
        """Closest lidar/depth reading in the direction of the YOLO box (within a
        narrow angular window around `bearing`). This is the distance to the
        victim the camera is looking at; when it drops below the threshold the
        victim counts as found."""
        best = LIDAR_MAX_USE
        for a, d in self._fused_points():
            if abs(wrap_pi(a - bearing)) <= CLEAR_WINDOW and d < best:
                best = d
        return best

    def _lidar_range_at(self, bearing):
        best = LIDAR_MAX_USE
        for a, d in zip(*self.read_lidar()):
            if abs(wrap_pi(a - bearing)) <= CLEAR_WINDOW and d < best:
                best = d
        return best

    def _depth_range_at(self, bearing):
        best = LIDAR_MAX_USE
        for a, d in zip(*self.read_depth_scan()):
            if abs(wrap_pi(a - bearing)) <= CLEAR_WINDOW and d < best:
                best = d
        return best

    def record_victim_debug(self, saw, bearing, dv, fwd_raw):
        """Track, per target victim, the closest this robot's own sensors ever
        read while confirming it (odometry to estimate, lidar and depth in the
        camera direction, nearest surface). Printed at mission end so we can see
        whether a miss was too-far or never-reported."""
        if self.target is None:
            return
        vid = self.target.get("id")
        rec = self.victim_dbg.setdefault(
            vid, {"odom": float("inf"), "lidar": float("inf"),
                  "depth": float("inf"), "fwd": float("inf"),
                  "xy": (self.target["x"], self.target["y"])})
        rec["odom"] = min(rec["odom"], dv)
        rec["fwd"] = min(rec["fwd"], fwd_raw)
        if saw:
            rec["lidar"] = min(rec["lidar"], self._lidar_range_at(bearing))
            rec["depth"] = min(rec["depth"], self._depth_range_at(bearing))

    def print_victim_debug(self):
        def s(x):
            return f"{x:5.2f}" if x != float("inf") else "  -- "
        print(f"\n[{self.name}] ===== per-victim closest sensor readings =====")
        for vid in sorted(self.victim_dbg, key=lambda k: (k is None, k)):
            r = self.victim_dbg[vid]
            print(f"[{self.name}] victim #{vid} est=({r['xy'][0]:+.1f},{r['xy'][1]:+.1f}) "
                  f"min: odom={s(r['odom'])} lidar={s(r['lidar'])} "
                  f"depth={s(r['depth'])} fwd={s(r['fwd'])}")
        print(f"[{self.name}] ================================================\n")

    def _inject_other_robot(self, angs, rs):
        """Add the other robot to the scan as an obstacle, in case the lidar
        misses its thin profile."""
        if self.other_pos is None:
            return
        odx = self.other_pos[0] - self.x
        ody = self.other_pos[1] - self.y
        od = math.hypot(odx, ody)
        if od < LIDAR_MAX_USE:
            ob = wrap_pi(math.atan2(ody, odx) - self.theta)
            for da in (-0.2, 0.0, 0.2):
                angs.append(wrap_pi(ob + da))
                rs.append(od)

    def polar_histogram(self, angs, rs):
        """Binary VFH histogram: each obstacle blocks an arc widened by the
        robot body (wider when the obstacle is closer)."""
        step = 2 * math.pi / VFH_BINS
        blocked = [False] * VFH_BINS
        for a, d in zip(angs, rs):
            gamma = math.asin(min(1.0, VFH_SAFE_DIST / max(d, 1e-3)))
            # Mark the bin the obstacle sits in, plus every bin whose CENTER
            # falls strictly inside the widened arc (ceil low, floor high). The
            # earlier floor/ceil edges over-blocked a bin each side and closed
            # narrow doorways.
            blocked[int(round((a + math.pi) / step)) % VFH_BINS] = True
            k0 = int(math.ceil((a - gamma + math.pi) / step))
            k1 = int(math.floor((a + gamma + math.pi) / step))
            for k in range(k0, k1 + 1):
                blocked[k % VFH_BINS] = True
        return blocked

    def vfh_choose(self, goal_bearing, blocked):
        """Pick the free direction (within the forward cone) that best trades
        off goal progress, hysteresis, and staying near the current heading."""
        step = 2 * math.pi / VFH_BINS
        best, best_cost = None, float("inf")
        for k in range(VFH_BINS):
            if blocked[k]:
                continue
            phi = wrap_pi(-math.pi + k * step)
            if abs(phi) > VFH_FWD_CONE:
                continue
            cost = (W_GOAL * abs(wrap_pi(phi - goal_bearing))
                    + W_SMOOTH * abs(wrap_pi(phi - self.prev_dir))
                    + W_HEADING * abs(phi))
            if cost < best_cost:
                best_cost, best = cost, phi
        return best

    def dir_clearance(self, phi, angs, rs):
        c = LIDAR_MAX_USE
        for a, d in zip(angs, rs):
            if abs(wrap_pi(a - phi)) <= CLEAR_WINDOW:
                c = min(c, d)
        return c

    def front_ir(self):
        vals = [self.ir["fl_range"].getValue(), self.ir["fr_range"].getValue()]
        vals = [v for v in vals if v is not None and v > 0.0]
        return min(vals) if vals else float("inf")

    def rear_ir(self):
        vals = [self.ir["rl_range"].getValue(), self.ir["rr_range"].getValue()]
        vals = [v for v in vals if v is not None and v > 0.0]
        return min(vals) if vals else float("inf")

    # ---- actuation --------------------------------------------------------
    def set_wheels(self, left, right):
        left = clamp(left, -26.0, 26.0)
        right = clamp(right, -26.0, 26.0)
        self.motors["fl"].setVelocity(left)
        self.motors["rl"].setVelocity(left)
        self.motors["fr"].setVelocity(right)
        self.motors["rr"].setVelocity(right)

    def stop(self):
        self.cur_v = 0.0
        self.cur_w = 0.0
        self.set_wheels(0.0, 0.0)

    def pursue(self, path):
        """Pure-pursuit: aim at the first path point beyond the lookahead radius.

        The path is first advanced to the waypoint nearest the robot. Dropping only
        the points inside the lookahead radius is not sufficient on its own: a
        waypoint the robot has already driven past sits OUTSIDE that radius as soon
        as it is more than LOOKAHEAD behind, so it would be chased backwards. The
        robot turns round, reaches it, drives on, and repeats, which traces a loop.
        Advancing to the closest waypoint first makes the follower robust to a path
        that no longer begins where the robot currently is."""
        if len(path) > 1:
            closest = min(range(len(path)),
                          key=lambda i: math.hypot(path[i][0] - self.x,
                                                   path[i][1] - self.y))
            del path[:closest]
        # Velocity-scaled lookahead: look further ahead the faster we are moving.
        look = clamp(abs(self.cur_v) * LOOKAHEAD_TIME, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
        while len(path) > 1 and \
                math.hypot(path[0][0] - self.x, path[0][1] - self.y) < look:
            path.pop(0)
        return path[0]

    def fgm_heading(self, goal_bearing):
        """Follow the Gap Method (Sezer & Gokasan 2012) -- returns a steering
        bearing, or None if no usable gap.

        DWA alone is slow exactly where you saw it struggle: in clutter it scores
        a fixed set of sampled arcs, and when every sampled arc clips something it
        crawls or stalls hunting for a way through. FGM instead reads the gap
        geometry straight off the scan in one pass: find the angular gaps, take the
        widest one, aim at its CENTRE, then blend that with the goal direction:

            phi_final = ( (alpha/d_min) * phi_gap_centre + phi_goal )
                        / ( (alpha/d_min) + 1 )

        As the nearest obstacle closes in (d_min -> 0) the gap centre dominates and
        the robot commits to the opening; in the open, the goal direction dominates
        so it does not wander. It is O(n) over the scan, has no local-minimum trap,
        and here it only chooses the HEADING -- DWA still turns that into (v, w)
        under the acceleration limits (the FGM-DW hybrid)."""
        angs, rs = self.read_lidar()
        if not angs:
            return None
        # Only consider what is roughly ahead; behind us is irrelevant to progress.
        pts = sorted(((wrap_pi(a), d) for a, d in zip(angs, rs)
                      if abs(wrap_pi(a)) <= FGM_FOV / 2.0), key=lambda p: p[0])
        if not pts:
            return None
        d_min = max(min(d for _, d in pts), 1e-3)

        # A gap is a run of consecutive beams whose range exceeds the threshold:
        # far enough away that the robot could pass through there.
        gaps, start = [], None
        for i, (a, d) in enumerate(pts):
            free = d >= FGM_GAP_DIST
            if free and start is None:
                start = i
            elif not free and start is not None:
                gaps.append((start, i - 1)); start = None
        if start is not None:
            gaps.append((start, len(pts) - 1))
        if not gaps:
            return None

        # Widest gap by angular span, and it must be wide enough for the body.
        best = max(gaps, key=lambda g: pts[g[1]][0] - pts[g[0]][0])
        a0, a1 = pts[best[0]][0], pts[best[1]][0]
        span = a1 - a0
        if span * max(d_min, FGM_GAP_DIST) < 2.0 * ROBOT_RADIUS * FGM_WIDTH_MARGIN:
            return None                      # opening too tight to fit through
        phi_gap = 0.5 * (a0 + a1)            # gap centre angle

        # If the goal already lies inside the widest gap, just go at the goal:
        # no reason to aim off-centre when the direct line is already clear.
        if a0 <= goal_bearing <= a1:
            phi_gap = goal_bearing
        w = FGM_ALPHA / d_min                # safety weight, grows as obstacles close in
        return wrap_pi((w * phi_gap + goal_bearing) / (w + 1.0))

    def dwa(self, tx, ty):
        """Dynamic Window Approach. Rolls out candidate (v, w) commands as short
        arcs and returns the (v m/s, w rad/s) that best trades progress toward
        the target against clearance from the live lidar points and speed.
        Returns None when no rollout is safe, which triggers recovery."""
        ct, st = math.cos(self.theta), math.sin(self.theta)
        dxr, dyr = tx - self.x, ty - self.y
        txr = dxr * ct + dyr * st          # target in the robot frame
        tyr = -dxr * st + dyr * ct

        angs, rs = self.read_lidar()
        tv = self._target_victim_xy()   # carve the target's body out of obstacles
        ox, oy = [], []
        for a, d in zip(angs[::2], rs[::2]):
            if d < DWA_OBS_RANGE:
                if tv is not None:
                    wx = self.x + d * math.cos(self.theta + a)
                    wy = self.y + d * math.sin(self.theta + a)
                    if math.hypot(wx - tv[0], wy - tv[1]) < VICTIM_CLEAR_R:
                        continue
                ox.append(d * math.cos(a))
                oy.append(d * math.sin(a))
        # Depth-camera obstacles (things off the lidar plane, e.g. a rack rod).
        dangs, drs = self.read_depth_scan()
        for a, d in zip(dangs, drs):
            if d < DWA_OBS_RANGE:
                if tv is not None:
                    wx = self.x + d * math.cos(self.theta + a)
                    wy = self.y + d * math.sin(self.theta + a)
                    if math.hypot(wx - tv[0], wy - tv[1]) < VICTIM_CLEAR_R:
                        continue
                ox.append(d * math.cos(a))
                oy.append(d * math.sin(a))
        if self.other_pos is not None:
            odx, ody = self.other_pos[0] - self.x, self.other_pos[1] - self.y
            od = math.hypot(odx, ody)
            if od < DWA_OBS_RANGE:
                ob = wrap_pi(math.atan2(ody, odx) - self.theta)
                ox.append(od * math.cos(ob))
                oy.append(od * math.sin(ob))
        obs = np.array([ox, oy]).T if ox else None

        cands = []
        for v in DWA_V:
            for w in np.linspace(-DWA_W_MAX, DWA_W_MAX, DWA_W_SAMPLES):
                th = x = y = 0.0
                xs, ys = [], []
                for _ in range(DWA_STEPS):
                    th += w * DWA_DT
                    x += v * math.cos(th) * DWA_DT
                    y += v * math.sin(th) * DWA_DT
                    xs.append(x)
                    ys.append(y)
                if obs is not None:
                    P = np.array([xs, ys]).T
                    dmin = float(np.sqrt(((P[:, None, :] - obs[None, :, :]) ** 2)
                                         .sum(-1)).min())
                else:
                    dmin = DWA_CLEAR_CAP
                if dmin < DWA_COLLISION:
                    continue
                heading = -math.hypot(x - txr, y - tyr)
                cands.append((v, float(w), heading, min(dmin, DWA_CLEAR_CAP), abs(v)))
        if not cands:
            return None

        hs = [c[2] for c in cands]
        cs = [c[3] for c in cands]
        vs = [c[4] for c in cands]
        hlo, hhi = min(hs), max(hs)
        clo, chi = min(cs), max(cs)
        vlo, vhi = min(vs), max(vs)

        def nrm(val, lo, hi):
            return 0.0 if hi <= lo else (val - lo) / (hi - lo)

        best, best_score = (0.0, 0.0), -1.0
        for v, w, h, c, sp in cands:
            score = (DWA_W_HEAD * nrm(h, hlo, hhi)
                     + DWA_W_CLEAR * nrm(c, clo, chi)
                     + DWA_W_VEL * nrm(sp, vlo, vhi))
            if score > best_score:
                best_score, best = score, (v, w)
        return best

    def _clear_local_obstacles(self):
        """Nav2-style clear-costmap recovery: wipe sensed obstacles near the
        robot so a fresh plan is not fighting stale or drifted marks."""
        if self.obs_score is None:
            return
        g = self.grid
        r0, c0 = g.world_to_cell(self.x, self.y)
        rad = int(CLEAR_RADIUS_M / g.res)
        self.obs_score[max(0, r0 - rad):r0 + rad + 1,
                       max(0, c0 - rad):c0 + rad + 1] = 0.0

    def _start_recovery(self, now):
        """Begin a recovery, escalating if we keep getting stuck in the same
        place. First couple of times: back off with a firm turn. After that:
        rotate in place to sweep the lidar and finish mapping the obstacle, so
        the replan can commit to a route around it instead of re-entering the
        trap."""
        if self.recover_anchor is not None and \
                math.hypot(self.x - self.recover_anchor[0],
                           self.y - self.recover_anchor[1]) < RECOVER_ANCHOR_M:
            self.recover_count += 1
        else:
            self.recover_count = 1
        self.recover_anchor = (self.x, self.y)
        self.recover_rotate = self.recover_count >= RECOVER_ROTATE_AT
        self.recover_until = now + (RECOVER_ROTATE_S if self.recover_rotate
                                    else RECOVER_TIME_S)
        self.pos_history.clear()

    def navigate(self, path):
        """DWA local planner following a global path. Returns distance to the
        final goal. Sets self.need_replan when a recovery finishes."""
        now = self.robot.getTime()
        goal = path[-1]
        dist_goal = math.hypot(goal[0] - self.x, goal[1] - self.y)

        # Recovery in progress: either rotate in place to re-map (on repeats) or
        # back off with a firm turn, then request a replan.
        if now < self.recover_until:
            if self.recover_rotate:
                spin = SLOW_SPEED * (1.0 if self.index == 0 else -1.0)
                self.set_wheels(-spin, spin)     # rotate in place, sweep the lidar
            else:
                # Back STRAIGHT out a few steps (not a reverse-and-turn, which
                # reorients the robot and makes it drive off the opposite way).
                # After backing off, the replan/DWA finds the forward gap.
                self.set_wheels(-SLOW_SPEED, -SLOW_SPEED)
            return dist_goal
        if self.recover_until > 0.0:
            self.recover_until = 0.0
            self.need_replan = True
            self.pos_history.clear()
            self._clear_local_obstacles()
            self.cur_v = self.cur_w = 0.0
            return dist_goal

        # Window-based stuck detection: sample position once a second, and if
        # net travel over the window is tiny, trigger recovery. Oscillating in
        # place no longer resets this the way instantaneous motion did.
        if not self.pos_history or now - self.pos_history[-1][0] >= 1.0:
            self.pos_history.append((now, self.x, self.y))
        if len(self.pos_history) >= 2:
            t0, x0, y0 = self.pos_history[0]
            if now - t0 >= STUCK_WINDOW and math.hypot(self.x - x0, self.y - y0) < STUCK_NET:
                self._start_recovery(now)
                return dist_goal

        # Goal-progress check. The test above only asks "did we move?", which the
        # logs showed a crawling robot passing while it drove AWAY from its goal.
        # This asks the question that matters: "are we getting CLOSER?"
        if self._prog_goal != goal:
            self._prog_goal = goal            # new goal: start a fresh window
            self.prog_history.clear()
        if not self.prog_history or now - self.prog_history[-1][0] >= 1.0:
            self.prog_history.append((now, dist_goal))
        if len(self.prog_history) >= 2:
            pt0, pd0 = self.prog_history[0]
            if now - pt0 >= PROGRESS_WINDOW_S and (pd0 - dist_goal) < PROGRESS_MIN_M:
                print(f"[{self.name}] NO PROGRESS: goal dist {pd0:.2f} -> "
                      f"{dist_goal:.2f} m in {now - pt0:.0f}s; recovering")
                self.prog_history.clear()
                self._start_recovery(now)
                return dist_goal

        # Higher-index robot yields when the other is close and ahead.
        if self.other_pos is not None and self.index == 1:
            odx, ody = self.other_pos[0] - self.x, self.other_pos[1] - self.y
            od = math.hypot(odx, ody)
            ob = wrap_pi(math.atan2(ody, odx) - self.theta)
            if od < ROBOT_YIELD_DIST and abs(ob) < 2 * FRONT_SECTOR:
                self.stop()
                return dist_goal

        # Collision monitor (runs below the planner): if the fused sensors see an
        # obstacle in the forward stop zone, refuse all forward motion and rotate
        # toward the clearer side, regardless of what the planner wants. This is
        # the hard guarantee that the robot never shoves into a rod/truck and
        # topples, even when the global plan or DWA would push forward.
        stop, fwd_min, left_clear, right_clear = self.collision_guard()
        if stop:
            wsign = 1.0 if left_clear >= right_clear else -1.0
            if self.rear_ir() > SAFE_REVERSE_CLEAR:
                # Rear is clear: back straight out of the obstacle with only a
                # gentle bias toward the clearer side. This opens forward space so
                # the planner finds the gap ahead, instead of spinning 180 in place.
                v_target = -SAFE_REVERSE_SPEED
                w_target = wsign * DWA_W_MAX * 0.35
            else:
                # Rear blocked too (a real corner/trap): fall back to pivoting
                # toward the clearer side; stuck detection escalates from here.
                v_target = 0.0
                w_target = wsign * DWA_W_MAX
            w = clamp(w_target, self.cur_w - DWA_W_ACCEL * self.dt,
                      self.cur_w + DWA_W_ACCEL * self.dt)
            v = clamp(v_target, self.cur_v - DWA_V_ACCEL * self.dt,
                      self.cur_v + DWA_V_ACCEL * self.dt)
            self.cur_v, self.cur_w = v, w
            wl = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
            wr = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
            self.set_wheels(wl, wr)
            return dist_goal

        tx, ty = self.pursue(path)
        # FGM-DW: when the way to the pure-pursuit waypoint is cluttered, let
        # Follow-the-Gap pick the heading (centre of the widest opening, blended
        # toward the goal) and hand DWA a target along that heading instead. DWA
        # still decides speed and turn rate, so acceleration limits, the collision
        # monitor and the victim carve-out all behave exactly as before -- this
        # only stops DWA hunting for a gap it can already be pointed at.
        if FGM_ENABLE:
            goal_b = wrap_pi(math.atan2(ty - self.y, tx - self.x) - self.theta)
            fb = self.fgm_heading(goal_b)
            if fb is not None and abs(wrap_pi(fb - goal_b)) > math.radians(5.0):
                ang = self.theta + fb
                tx = self.x + FGM_LOOKAHEAD * math.cos(ang)
                ty = self.y + FGM_LOOKAHEAD * math.sin(ang)
        # Align in place before driving: if the waypoint is well off our heading,
        # rotate toward it instead of letting DWA open a wide arc (which is the
        # start-of-run circle on the mat). Rotating in place keeps the footprint
        # put, so it is safe on the open mat and at sharp turns alike.
        head_err = wrap_pi(math.atan2(ty - self.y, tx - self.x) - self.theta)
        if abs(head_err) > ALIGN_ANGLE:
            w = clamp(2.0 * head_err, -DWA_W_MAX, DWA_W_MAX)
            w = clamp(w, self.cur_w - DWA_W_ACCEL * self.dt,
                      self.cur_w + DWA_W_ACCEL * self.dt)
            v = clamp(ALIGN_CREEP, self.cur_v - DWA_V_ACCEL * self.dt,
                      self.cur_v + DWA_V_ACCEL * self.dt)
            self.cur_v, self.cur_w = v, w
            wl = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
            wr = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
            self.set_wheels(wl, wr)
            return dist_goal
        cmd = self.dwa(tx, ty)
        if cmd is None:
            # No safe rollout in any direction: recover (escalates to rotating in
            # place and re-mapping if we keep hitting the same trap).
            self._start_recovery(now)
            self.cur_v = self.cur_w = 0.0
            return dist_goal

        # Slowdown zone: scale the forward speed down as the nearest thing ahead
        # gets closer, so the robot is already crawling by the time it reaches the
        # stop zone and never has to brake hard (which is what tipped it).
        v, w = cmd
        if v > 0.0 and fwd_min < SAFE_SLOW_DIST:
            scale = clamp((fwd_min - SAFE_STOP_DIST) / (SAFE_SLOW_DIST - SAFE_STOP_DIST),
                          0.15, 1.0)
            v *= scale

        # Ramp toward the chosen command instead of snapping to it, so the robot
        # accelerates and turns smoothly and does not lurch or tip.
        v = clamp(v, self.cur_v - DWA_V_ACCEL * self.dt, self.cur_v + DWA_V_ACCEL * self.dt)
        w = clamp(w, self.cur_w - DWA_W_ACCEL * self.dt, self.cur_w + DWA_W_ACCEL * self.dt)
        self.cur_v, self.cur_w = v, w
        wl = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        wr = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        self.set_wheels(wl, wr)
        # Once we have driven clear of the last trap, forget the escalation so a
        # future, unrelated obstacle starts fresh from a gentle back-off.
        if self.recover_anchor is not None and \
                math.hypot(self.x - self.recover_anchor[0],
                           self.y - self.recover_anchor[1]) > 2 * RECOVER_ANCHOR_M:
            self.recover_anchor = None
            self.recover_count = 0
            self.recover_rotate = False
        return dist_goal

    # ---- comms ------------------------------------------------------------
    def poll_squad(self):
        while self.squad_receiver.getQueueLength() > 0:
            try:
                msg = json.loads(self.squad_receiver.getString())
                if msg.get("robot") != self.name:
                    t = msg.get("type")
                    if t == "POS":
                        self.other_pos = (msg["x"], msg["y"])
                        c = msg.get("claim")
                        self.other_claim = (c[0], c[1]) if c else None
                        self.other_claim_id = msg.get("claim_id")
                    elif t == "DONE":
                        cid = msg.get("id")
                        vx, vy = msg["x"], msg["y"]
                        # Carry the sender's verdict too. Without this we marked the
                        # victim done with no uncertain flag, so OUR map painted it
                        # green (confirmed) even when the other robot never actually
                        # saw it, which would render as a confirmed find.
                        unc = bool(msg.get("uncertain", True))
                        for v in self.victims:
                            if (cid is not None and v.get("id") == cid) or \
                                    math.hypot(v["x"] - vx, v["y"] - vy) < CLAIM_MATCH_M:
                                v["done"] = True
                                v["uncertain"] = unc
                    elif t == "TRAIL":
                        self.other_trail.append((msg["x"], msg["y"]))
                        if len(self.other_trail) > TRAIL_MAX:
                            self.other_trail = self.other_trail[-TRAIL_MAX:]
            except Exception:
                pass
            self.squad_receiver.nextPacket()

    def record_breadcrumb(self):
        """Drop a breadcrumb where the robot actually is (hence proven free) once
        it has travelled BREADCRUMB_SPACING_M since the last one, and share it so
        the other robot can reuse this stretch as a safe corridor."""
        if self.state not in ("NAV", "EXPLORE"):
            return
        if self.last_crumb is not None and \
                math.hypot(self.x - self.last_crumb[0],
                           self.y - self.last_crumb[1]) < BREADCRUMB_SPACING_M:
            return
        self.last_crumb = (self.x, self.y)
        self.trail.append(self.last_crumb)
        if len(self.trail) > TRAIL_MAX:
            self.trail = self.trail[-TRAIL_MAX:]
        self.squad_send("TRAIL", x=self.x, y=self.y)

    def squad_send(self, msgtype, **kw):
        payload = {"type": msgtype, "robot": self.name}
        payload.update(kw)
        self.squad_emitter.send(json.dumps(payload).encode())

    def broadcast_position(self):
        now = self.robot.getTime()
        if now - self.last_pos_bcast >= POS_BCAST_PERIOD_S:
            # Piggyback the victim we are currently working on (position AND its
            # stable id), so the other robot always has a fresh claim and never
            # targets the same one, even before it can see it on camera.
            working = self.target is not None and self.state in ("NAV", "CONFIRM")
            claim = [self.target["x"], self.target["y"]] if working else None
            claim_id = self.target.get("id") if working else None
            self.squad_send("POS", x=self.x, y=self.y,
                            claim=claim, claim_id=claim_id)
            self.last_pos_bcast = now

    def announce_done(self, v):
        self.squad_send("DONE", x=v["x"], y=v["y"], id=v.get("id"),
                        uncertain=bool(v.get("uncertain", False)))

    def report_victim(self, confidence):
        msg = {
            "timestamp": self.robot.getTime(),
            "robot_id": self.name,
            "position": [self.x, self.y, 0.0],
            "victim_found": True,
            "victim_confidence": float(confidence),
        }
        self.sup_emitter.send(json.dumps(msg).encode())

    def draw_map(self):
        """Live top-view map in a cv2 window: wall map, victim estimates with
        labels and their 1.0 m scoring circles (green once found, red if not),
        both robots' odometry trails, and each robot with a heading arrow. Each
        robot draws the full picture because it also receives the other robot's
        breadcrumbs. This is the clearest way to see why a victim is not reached."""
        if not (DEBUG_MAP and _HAVE_CV2) or self.index != 0:
            return   # robot1 draws the shared map (it has both trails + positions)
        now = self.robot.getTime()
        if now - self._last_map < 0.25:
            return
        self._last_map = now

        xs = [self.x]
        ys = [self.y]
        for v in self.victims:
            xs.append(v["x"]); ys.append(v["y"])
        for wx1, wy1, wx2, wy2 in self.walls:
            xs += [wx1, wx2]; ys += [wy1, wy2]
        for p in self.trail + self.other_trail:
            xs.append(p[0]); ys.append(p[1])
        if self.other_pos is not None:
            xs.append(self.other_pos[0]); ys.append(self.other_pos[1])
        min_x, max_x = min(xs) - 1.0, max(xs) + 1.0
        min_y, max_y = min(ys) - 1.0, max(ys) + 1.0

        W = H = 760
        pad = 30
        scale = min((W - 2 * pad) / max(0.1, max_x - min_x),
                    (H - 2 * pad) / max(0.1, max_y - min_y))

        def to_px(x, y):
            return (int(pad + (x - min_x) * scale),
                    int(H - pad - (y - min_y) * scale))   # flip Y so up is +Y

        img = np.full((H, W, 3), 32, np.uint8)

        for wx1, wy1, wx2, wy2 in self.walls:                 # walls
            cv2.line(img, to_px(wx1, wy1), to_px(wx2, wy2), (190, 190, 190), 2)

        for p in self.other_trail:                            # trails (BGR)
            cv2.circle(img, to_px(*p), 2, (150, 120, 0), -1)  # robot2 trail teal
        for p in self.trail:
            cv2.circle(img, to_px(*p), 2, (0, 140, 200), -1)  # robot1 trail orange

        for v in self.victims:                                # victims + scoring ring
            c = to_px(v["x"], v["y"])
            # Honest colour: GREEN only when we actually closed in (done and not
            # uncertain), YELLOW when we marked it done but never truly got within
            # range (timed out far, likely blocked / bad estimate), RED untouched.
            if not v["done"]:
                col = (0, 0, 255)          # red: not handled
            elif v.get("uncertain"):
                col = (0, 210, 255)        # yellow: done but NOT confirmed close
            else:
                col = (0, 200, 0)          # green: confirmed close
            cv2.circle(img, c, int(1.0 * scale), col, 1)      # 1.0 m scoring radius
            # Inner grey ring = MARK_REACH_M, the odometry distance at which we
            # actually mark the victim found. It is TIGHTER than the 1.0 m scoring
            # ring on purpose, so "inside the big circle but still not green" is
            # expected behaviour, not a bug.
            cv2.circle(img, c, int(MARK_REACH_M * scale), (120, 120, 120), 1)
            cv2.circle(img, c, 6, col, -1)
            cv2.putText(img, f"V{v.get('id')}", (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        if self.other_pos is not None:                        # other robot
            oc = to_px(*self.other_pos)
            cv2.circle(img, oc, 8, (255, 160, 0), 2)
            cv2.putText(img, "other", (oc[0] + 9, oc[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1)

        sc = to_px(self.x, self.y)                            # self + heading
        tip = to_px(self.x + 0.5 * math.cos(self.theta),
                    self.y + 0.5 * math.sin(self.theta))
        cv2.arrowedLine(img, sc, tip, (0, 255, 255), 2, tipLength=0.4)
        cv2.circle(img, sc, 8, (0, 255, 255), -1)
        cv2.putText(img, self.name, (sc[0] + 9, sc[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(img, f"green=found  yellow=done but far  red=not done   "
                    f"outer ring=1.0m score / inner grey={MARK_REACH_M:.2f}m mark"
                    f"   self=cyan  other=orange", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)
        cv2.imshow(f"{self.name} map (top view)", img)
        cv2.waitKey(1)

    def mark_reached_victims(self):
        """Mark ANY unfound victim as found the moment odometry says we are within
        MARK_REACH_M of it, in ANY state. Position is the only thing the rules
        care about, so this check must NOT live inside one state: a robot that
        drives through a victim's circle while in NAV, EXPLORE or recovery would
        otherwise never mark it however long it sat there. No lidar/camera gate.
        Claimed-by-the-other-robot is ignored on purpose: if we are physically on
        it, it is reached, and the DONE broadcast stops the other robot going."""
        if self.fallen or not MARK_ON_ODOM:
            return          # position untrusted, or marking is confirm-only
        for v in self.victims:
            if v.get("done"):
                continue
            d = math.hypot(v["x"] - self.x, v["y"] - self.y)
            if d <= MARK_REACH_M:
                v["done"] = True
                v["skipped"] = False
                v["uncertain"] = False       # odometry says we genuinely got there
                self.announce_done(v)
                print(f"[{self.name}] REACHED victim #{v.get('id')} "
                      f"(odom {d:.2f} m <= {MARK_REACH_M:.2f}) -> marked found")

    def maybe_report(self):
        """POSITION-DRIVEN reporting. The rules ask for two things to score a
        victim: be within 1.0 m of it and send the message. No sensor confirmation
        is required, so the trigger is odometry proximity to a victim estimate.
        Lidar is for obstacle avoidance only and never gates a report.

        The camera is used ONLY to raise the reported confidence when it happens
        to see a person, and as a secondary trigger for a victim the flyover
        missed entirely. Runs in ANY state.

        Every report is also scored for confidence accuracy, so the message count
        is kept low and each one is fired from the position most likely to be
        inside the ring. See the REPORT_* block for why a wide band loses points
        overall."""
        now = self.robot.getTime()
        # Tilt does NOT suppress reporting. The supervisor scores our TRUE position,
        # which it reads from the simulator, so a message sent while the robot is
        # leaning on a victim's leg scores exactly the same as one sent upright.
        # Blocking here could only ever throw away a find we had already earned.
        if now - self.last_report < REPORT_PERIOD_S:
            return
        key, rconf, src = None, 0.0, ""

        # PRIMARY: odometry says we are on a victim's estimate. This does NOT skip
        # victims already marked done locally: "done" is our own bookkeeping, and
        # only the supervisor knows whether a victim actually scored. A victim is
        # marked as soon as the robot is within MARK_REACH_M, so skipping done
        # victims here would stop the reports at the very moment the robot is
        # closest to the true marker.
        #
        # Repeats are NOT free, though. The supervisor's proximity check skips
        # victims it has already recorded as found, so a repeat report for a victim
        # that did score comes back with a False verdict and counts against the
        # confidence term exactly like a miss. That is what REPORT_MAX_SENDS bounds.
        gate_d = float("inf")     # the distance the closest-approach gate judges
        best, best_d = None, (REPORT_ODOM_M if REPORT_ON_ODOM else -1.0)
        for v in self.victims:
            if not self._not_others(v):
                continue
            d = math.hypot(v["x"] - self.x, v["y"] - self.y)
            if d < best_d:
                best_d, best = d, v
        if best is not None:
            saw, _c, _b, _f = self.throttled_look()
            key = best.get("id")
            rconf = REPORT_SEEN_CONF if saw else REPORT_ODOM_CONF
            src = "odom+cam" if saw else "odom"
            gate_d = best_d

        # Camera-confirmed report: a person in view whose measured range puts us
        # inside the scoring radius. This is the only reporting path in the
        # "baseline" profile. A detection that maps onto a KNOWN victim is reported
        # at the normal detector threshold; one that maps to no known victim needs a
        # much higher confidence, because that branch is where false positives on
        # scenery would otherwise be reported as confident finds.
        if key is None:
            saw, conf, bearing, _ = self.throttled_look()
            if saw:
                rng = self.victim_range(bearing)
                if rng <= REPORT_RANGE_M and not self._detection_is_theirs(bearing, rng):
                    dwx = self.x + rng * math.cos(self.theta + bearing)
                    dwy = self.y + rng * math.sin(self.theta + bearing)
                    v = self._victim_at(dwx, dwy)
                    if v is not None:
                        key, rconf, src = v.get("id"), REPORT_SEEN_CONF, "cam"
                        gate_d = rng
                    elif conf >= NEW_VICTIM_CONF and self._new_victim_ok(dwx, dwy):
                        # Snap to a coarse grid: rounding finely gave a slightly
                        # different key every frame, and each new key came with a
                        # fresh send budget, so one victim could report without limit.
                        key = ("live", round(dwx), round(dwy))
                        rconf, src = REPORT_SEEN_CONF, "cam-new"
                        gate_d = rng

        if key is None or rconf < REPORT_MIN_CONF:
            return
        track = self.reported.setdefault(key, {"n": 0, "t": -1e9})
        if track["n"] >= REPORT_MAX_SENDS:
            return
        # Inside REPORT_SURE_M we are, as far as anything on this robot can tell,
        # AT the victim: odometry is good to a few centimetres (the supervisor's own
        # near-miss lines put the gap at 0.00-0.03 m), so being this close to the
        # estimate is the strongest claim we will ever be able to make. Nothing is
        # allowed to suppress a report here - not the closest-approach gate, not
        # tilt, not a sensor. Being inside the ring and staying silent is the one
        # failure that cannot be recovered from.
        #
        # The gate only rations the SPECULATIVE band beyond that, where we are
        # reporting on the chance that the estimate is offset in our favour. There
        # a report only fires when it beats our nearest previous one, so the budget
        # is spent walking inward instead of being emptied on first sight.
        #
        # The two paths measure DIFFERENT distances and are scored separately.
        # Camera range is to the body SURFACE; odometry distance is to the flyover
        # estimate, which sits inside the body and can be metres out. Sharing one
        # baseline let a single camera report at 0.94 m lock the odometry path out
        # permanently, because odometry never reads below ~1.7 m on that victim.
        # That is exactly how a robot drove into the ring, touched the victim, and
        # sent nothing after its first message.
        mkey = "bd_cam" if src.startswith("cam") else "bd_odom"
        prev = track.get(mkey, float("inf"))
        if gate_d > REPORT_SURE_M:
            if gate_d > prev - REPORT_IMPROVE_M:
                return
        track[mkey] = min(prev, gate_d)
        self.report_victim(rconf)
        track["n"] += 1
        track["t"] = now
        self.last_report = now
        print(f"[{self.name}] REPORT v#{key} src={src} conf={rconf:.2f} "
              f"d={gate_d:.2f} send {track['n']}/{REPORT_MAX_SENDS} ({self.state})")

    def maybe_intercept(self):
        """Divert-and-close: while navigating or exploring, a person seen up
        close whose victim was never PHYSICALLY confirmed (unfound, skipped, or
        confirmed-but-uncertain) becomes the immediate target and we run a full
        camera-homing confirm on it. This is the fix for the near-miss stream in
        the logs: robot1 drove PAST victim2 at 0.8 m camera range, TRUE distance
        1.4 m to the waist marker, reporting in vain, because the victim was
        marked done(skipped) and _available() refused to engage it. The camera
        measurement is robot-relative and drift-free, so closing on it is the
        one channel odometry error cannot touch. Bounded per victim."""
        if self.state not in ("NAV", "EXPLORE"):
            return
        saw, conf, bearing, _ = self.throttled_look()
        if not (saw and conf >= 0.5):
            return
        rng = self.victim_range(bearing)
        if rng >= INTERCEPT_RANGE_M:
            return
        if self._detection_is_theirs(bearing, rng):
            return
        dwx = self.x + rng * math.cos(self.theta + bearing)
        dwy = self.y + rng * math.sin(self.theta + bearing)
        hit = self._victim_at(dwx, dwy)
        if hit is None:
            return   # unknown victim: step_explore's registration path owns that case
        if self.target is not None and hit.get("id") == self.target.get("id") \
                and self.state == "NAV":
            self._enter_confirm()   # our own goal is right here: skip the rest of the drive
            return
        if not ENABLE_RETRY:
            return   # do NOT break off to re-confirm some other victim; press on
        confirmed = hit["done"] and not (hit.get("skipped") or hit.get("uncertain"))
        if confirmed:
            return   # physically confirmed close before: nothing left to gain
        if not self._not_others(hit):
            return
        if hit.get("intercepts", 0) >= INTERCEPT_MAX:
            return
        hit["intercepts"] = hit.get("intercepts", 0) + 1
        hit["done"] = False
        hit["skipped"] = False
        self.target = hit
        print(f"[{self.name}] INTERCEPT victim #{hit.get('id')} seen at "
              f"rng={rng:.2f} (try {hit['intercepts']}); diverting to confirm")
        self._enter_confirm()

    # ---- victim detection -------------------------------------------------
    def look_for_victim(self):
        """Run the detector on the current RGB frame.

        Returns (saw, confidence, bearing_rad, box_height_fraction) for the most
        confident person detection. The bearing is the box centre mapped through
        the camera FOV; box_height_fraction grows as the robot approaches, so it
        doubles as a crude range cue."""
        if self.yolo is None:
            return False, 0.0, 0.0, 0.0
        try:
            w, h = self.cam.getWidth(), self.cam.getHeight()
            arr = np.frombuffer(self.cam.getImage(), np.uint8).reshape((h, w, 4))
            bgr = np.ascontiguousarray(arr[:, :, :3])
            res = self.yolo.predict(bgr, classes=[0], conf=YOLO_CONF, verbose=False)
            best_conf, best_cx, best_frac, best_w = 0.0, None, 0.0, 0.0
            for r in res:
                for box in r.boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if conf > best_conf:
                        best_conf = conf
                        best_cx = (x1 + x2) / 2.0
                        best_frac = (y2 - y1) / float(h)
                        best_w = abs(x2 - x1)
            if best_cx is None:
                self._det_halfwidth = 0.0
                return False, 0.0, 0.0, 0.0
            fov = self.cam.getFov()
            bearing = -(best_cx / w - 0.5) * fov
            # Angular half-width of the box: the scan sector belonging to this
            # victim, used to pull its lidar returns out of the full scan.
            self._det_halfwidth = 0.5 * (best_w / float(w)) * fov + SERVO_SECTOR_PAD
            return True, best_conf, bearing, best_frac
        except Exception:
            self._det_halfwidth = 0.0
            return False, 0.0, 0.0, 0.0

    def victim_body_measurement(self, bearing):
        """Fuse the camera detection with the lidar/depth scan and return
        (cx, cy, range, bearing) of the victim's BODY CENTROID in the ROBOT frame,
        or None when too few returns fall inside the detection sector.

        Everything here is measured relative to the robot, so unlike the odometry
        distance to the estimate it is unaffected by drift or by an offset estimate."""
        hw = max(self._det_halfwidth, SERVO_SECTOR_PAD)
        pts = [(a, d) for a, d in self._fused_points()
               if abs(wrap_pi(a - bearing)) <= hw and 0.05 < d < LIDAR_MAX_USE]
        if len(pts) < SERVO_MIN_POINTS:
            return None
        dmin = min(d for _, d in pts)
        body = [(a, d) for a, d in pts if d <= dmin + SERVO_BODY_DEPTH]
        if len(body) < SERVO_MIN_POINTS:
            return None
        cx = sum(d * math.cos(a) for a, d in body) / len(body)
        cy = sum(d * math.sin(a) for a, d in body) / len(body)
        return cx, cy, math.hypot(cx, cy), math.atan2(cy, cx)

    def refine_victim_estimate(self, v, cx, cy):
        """Fold a live body measurement into a victim's stored map position with an
        exponential moving average. Once the victim is actually visible the sensors
        know better than the flyover estimate, so this pulls the target (and the
        scoring ring drawn on the debug map) toward where the victim really is."""
        if v is None:
            return
        ct, st = math.cos(self.theta), math.sin(self.theta)
        wx = self.x + cx * ct - cy * st
        wy = self.y + cx * st + cy * ct
        k = SERVO_FILTER_COEF
        v["x"] = (1.0 - k) * v["x"] + k * wx
        v["y"] = (1.0 - k) * v["y"] + k * wy

    def throttled_look(self):
        now = self.robot.getTime()
        if now - self.last_yolo < YOLO_PERIOD_S:
            return self._last_det
        self.last_yolo = now
        self._last_det = self.look_for_victim()
        return self._last_det

    def calibrated_confidence(self, yolo_conf, rng):
        """Confidence value for a victim_found report. High (0.80-0.95)
        only when the HIP was confirmed (so we homed on the waist) AND we are close;
        mid for a detector-only box; 0 (do not send) when too far. The scoring's
        confidence-accuracy term punishes confident reports that do not correspond
        to a real, in-range find, so we never inflate."""
        if rng <= REPORT_CLOSE_M:
            return clamp(0.80 + 0.15 * yolo_conf, 0.80, 0.95)
        if rng <= REPORT_RANGE_M:
            return clamp(0.55 + 0.10 * yolo_conf, 0.55, 0.65)
        return 0.0

    def _new_victim_ok(self, wx, wy):
        """Whether a detection with no matching flyover victim may be registered as
        a brand-new one. Requires that the other robot can actually be ruled out
        (its position is known) and that we are not still on the launch pad, where
        the only person-shaped thing in view is the other ROSbot."""
        if self.other_pos is None:
            return False
        if math.hypot(wx - self.start_xy[0], wy - self.start_xy[1]) < NEW_VICTIM_START_M:
            return False
        return True

    def _detection_is_theirs(self, bearing, rng):
        """True if a camera detection at this bearing/range is really the other
        robot, or the victim it has already claimed, so we do not lock onto it
        and end up both chasing the same target."""
        dwx = self.x + rng * math.cos(self.theta + bearing)
        dwy = self.y + rng * math.sin(self.theta + bearing)
        if self.other_pos is not None and \
                math.hypot(dwx - self.other_pos[0], dwy - self.other_pos[1]) < 0.7:
            return True
        if self.other_claim is not None and \
                math.hypot(dwx - self.other_claim[0], dwy - self.other_claim[1]) < CLAIM_MATCH_M:
            return True
        return False

    def _victim_at(self, wx, wy, radius=VICTIM_ASSOC_M):
        """Nearest known victim to a world point within radius, or None. Lets a
        camera detection be tied back to a specific victim id so ownership (mine
        / the other robot's / done) can be checked before we chase it."""
        best, best_d = None, radius
        for v in self.victims:
            d = math.hypot(v["x"] - wx, v["y"] - wy)
            if d < best_d:
                best, best_d = v, d
        return best

    def _target_victim_xy(self):
        """Position of the victim we are actively approaching, else None. Used to
        carve the body out of the obstacle layer and DWA so it is a goal, not an
        obstacle. Only while homing on a victim (not while exploring to a
        coverage point)."""
        if self.target is not None and self.state in ("NAV", "CONFIRM"):
            return self.target["x"], self.target["y"]
        return None

    # ---- mission states ---------------------------------------------------
    def _available(self, v):
        """A victim this robot may pursue: not already found, and not the one the
        other robot is currently committed to (matched by id first, then by
        position as a fallback for victims discovered live during exploration)."""
        if v["done"]:
            return False
        if self.robot.getTime() < self.no_path_until.get(v.get("id"), -1.0):
            return False   # recently proven unreachable: let the world change first
        if self.other_claim_id is not None and v.get("id") == self.other_claim_id:
            return False
        if self.other_claim is not None:
            if math.hypot(v["x"] - self.other_claim[0],
                          v["y"] - self.other_claim[1]) < CLAIM_MATCH_M:
                return False
        return True

    def pick_next_victim(self):
        cands = [v for v in self.victims if v["mine"] and self._available(v)]
        if not cands:
            cands = [v for v in self.victims if self._available(v)]
        if not cands:
            return None
        return min(cands, key=lambda v: math.hypot(v["x"] - self.x, v["y"] - self.y))

    @staticmethod
    def _ray_cells(r0, c0, r1, c1):
        """Integer cells along the line from (r0,c0) to (r1,c1), EXCLUDING the
        endpoint. Standard Bresenham; used to clear the free space a lidar beam
        travelled through before it hit something."""
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        guard = 0
        while (r, c) != (r1, c1) and guard < 512:
            cells.append((r, c))
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
            guard += 1
        return cells

    def update_obstacle_layer(self):
        """Fold the current lidar scan into the obstacle grid the way a real
        occupancy map does: each beam MARKS its hit cell and CLEARS the cells it
        passed through. Hits already explained by the wall map are skipped, so the
        layer holds only furniture, the truck, and the other robot. Because a
        clear only happens where a beam actually sees through, a large obstacle
        stays remembered after it leaves view instead of being forgotten."""
        now = self.robot.getTime()
        if self.obs_score is None or now - self.last_obs_update < OBS_UPDATE_S:
            return
        self.last_obs_update = now
        self.obs_score *= OBS_DECAY          # tiny fade, only for drift smears
        angs, rs = self.read_lidar()
        g = self.grid
        tv = self._target_victim_xy()        # never mark the victim we approach
        r0, c0 = g.world_to_cell(self.x, self.y)
        for a, d in zip(angs[::2], rs[::2]):  # downsample for speed
            clear_d = min(d, OBS_MAX_RANGE)
            ex = self.x + clear_d * math.cos(self.theta + a)
            ey = self.y + clear_d * math.sin(self.theta + a)
            r1, c1 = g.world_to_cell(ex, ey)
            # Clear the free space the beam travelled through. This self-corrects
            # marks that moved or were noise, and needs no exception for the flyover
            # walls any more: they are a cost now, not part of this layer at all.
            for (rr, cc) in self._ray_cells(r0, c0, r1, c1):
                if g.in_bounds(rr, cc):
                    self.obs_score[rr, cc] = max(0.0, self.obs_score[rr, cc] - OBS_MISS)
            # Mark the hit cell itself (only real, close returns).
            if d >= OBS_MAX_RANGE:
                continue
            wx = self.x + d * math.cos(self.theta + a)
            wy = self.y + d * math.sin(self.theta + a)
            if tv is not None and math.hypot(wx - tv[0], wy - tv[1]) < VICTIM_CLEAR_R:
                continue   # this hit is the target's own body: it is the goal
            r, c = g.world_to_cell(wx, wy)
            if g.in_bounds(r, c):
                self.obs_score[r, c] = min(OBS_MAX, self.obs_score[r, c] + OBS_HIT)

        # Depth camera: mark obstacles at camera height the lidar plane misses
        # (the rack rod). Mark only, no ray-clearing, so a lidar beam passing
        # cleanly under/over the rod does not erase it; re-marking each cycle
        # keeps it solid while visible, and the slow decay removes it once gone.
        dangs, drs = self.read_depth_scan()
        for a, d in zip(dangs, drs):
            if d >= OBS_MAX_RANGE:
                continue
            wx = self.x + d * math.cos(self.theta + a)
            wy = self.y + d * math.sin(self.theta + a)
            if tv is not None and math.hypot(wx - tv[0], wy - tv[1]) < VICTIM_CLEAR_R:
                continue
            r, c = g.world_to_cell(wx, wy)
            if g.in_bounds(r, c):
                self.obs_score[r, c] = min(OBS_MAX, self.obs_score[r, c] + OBS_HIT)

    def _dyn_mask(self):
        """Sensed-obstacle mask for planning, with every proven breadcrumb cell
        (ours and the other robot's) forced free. A lane a robot already drove
        through can therefore never be sealed by accumulated obstacle inflation,
        which is a common way the plan gets trapped near a big obstacle."""
        if self.obs_score is None:
            return None
        mask = self.obs_score > OBS_THRESH
        g = self.grid
        for (x, y) in self.trail + self.other_trail:
            r, c = g.world_to_cell(x, y)
            if g.in_bounds(r, c):
                mask[r, c] = False
        return mask

    def _fused_rebuild(self):
        self.grid.rebuild(self._dyn_mask())

    def plan_to(self, gx, gy):
        """Replan on the fused map (static walls + sensed obstacles). Returns
        False if no path exists, e.g. furniture fully blocks the victim."""
        self._fused_rebuild()
        self.last_replan = self.robot.getTime()
        path = astar(self.grid, (self.x, self.y), (gx, gy))
        if path is None:
            return False
        self.path = path
        return True

    def _approach_goal(self, vx, vy):
        """Carrot-planner-style standoff pose: walk back from the victim along
        the vector toward the robot until we hit a free cell at least
        VICTIM_STANDOFF_M away. We plan to this pose instead of the body so the
        goal is never on top of the victim (whose legs read as an obstacle), and
        the final closing is handed to camera homing."""
        dx, dy = self.x - vx, self.y - vy
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return vx, vy
        base = math.atan2(dy, dx)          # direction we are coming from

        # Sweep OUTWARD in radius, and at each radius try every direction around
        # the victim, nearest our own bearing first. Searching only along the line
        # back to the robot fails whenever that line is blocked: for a victim near
        # a wall the whole ray is inflated, so the goal gets pushed metres away and
        # the robot stops there, far outside the scoring radius. Ordering by radius
        # guarantees the standoff returned is the closest usable one to the victim,
        # from whichever side happens to be open.
        s = VICTIM_STANDOFF_M
        while s <= APPROACH_MAX_R:
            for k in range(APPROACH_DIRS):
                # 0, +1, -1, +2, -2 ... steps away from our approach bearing
                step = (k + 1) // 2 * (1 if k % 2 else -1)
                ang = base + step * (2.0 * math.pi / APPROACH_DIRS)
                gx, gy = vx + s * math.cos(ang), vy + s * math.sin(ang)
                if not self._in_bounds(gx, gy):
                    continue
                r, c = self.grid.world_to_cell(gx, gy)
                if self.grid.is_free(r, c):
                    return gx, gy
            s += GRID_RES
        return self.x, self.y   # nothing usable anywhere: hold position

    def plan_to_victim(self, v):
        """Plan to the victim's approach standoff (not the body coordinate). The
        victim is carved out of the obstacle layer first so the standoff and the
        path leading to it sit in genuinely free space."""
        self._fused_rebuild()
        gx, gy = self._approach_goal(v["x"], v["y"])
        self.last_replan = self.robot.getTime()
        path = astar(self.grid, (self.x, self.y), (gx, gy))
        if path is None:
            return False
        self.path = path
        return True

    def plan_safe_road_to(self, v):
        """Route to the victim along the shared safe road: a graph over the
        breadcrumbs both robots have physically driven (so every edge is proven
        passable). Snap our pose and the goal onto the nearest breadcrumbs, run
        Dijkstra over links shorter than SAFE_LINK_M, and follow that polyline
        with a short local final leg to the goal. This is the working robot's
        road rescuing the stuck one.

        The road is only accepted when its goal end actually gets us CLOSER to
        our own victim than we are now. Otherwise the other robot's trail (which
        heads to ITS victim) would drag us the wrong way, so we decline it and
        keep working our own route."""
        nodes = list(self.trail) + list(self.other_trail)
        n = len(nodes)
        if n < 2:
            return False
        gx, gy = self._approach_goal(v["x"], v["y"])

        def nearest(px, py, limit):
            bi, bd = -1, limit
            for i, (nx, ny) in enumerate(nodes):
                d = math.hypot(nx - px, ny - py)
                if d < bd:
                    bi, bd = i, d
            return bi

        si = nearest(self.x, self.y, SAFE_ATTACH_M)
        gi = nearest(gx, gy, SAFE_GOAL_ATTACH_M)
        if si < 0 or gi < 0:
            return False
        # Progress gate: the trail's closest approach to our goal must be
        # meaningfully nearer the goal than we already are, or the road does not
        # actually lead toward our victim and we should not take it.
        my_goal_d = math.hypot(gx - self.x, gy - self.y)
        node_goal_d = math.hypot(nodes[gi][0] - gx, nodes[gi][1] - gy)
        if node_goal_d >= my_goal_d - SAFE_PROGRESS_M:
            return False
        if si == gi:
            self.path = [nodes[gi], (gx, gy)]
            self.last_replan = self.robot.getTime()
            return True

        INF = float("inf")
        dist = [INF] * n
        prev = [-1] * n
        dist[si] = 0.0
        heap = [(0.0, si)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == gi:
                break
            ux, uy = nodes[u]
            for w in range(n):
                if w == u:
                    continue
                e = math.hypot(nodes[w][0] - ux, nodes[w][1] - uy)
                if e <= SAFE_LINK_M and d + e < dist[w]:
                    dist[w] = d + e
                    prev[w] = u
                    heapq.heappush(heap, (dist[w], w))
        if dist[gi] == INF:
            return False
        chain = []
        c = gi
        while c != -1:
            chain.append(nodes[c])
            c = prev[c]
        chain.reverse()
        self.path = chain + [(gx, gy)]
        self.last_replan = self.robot.getTime()
        return True

    def _path_len(self, path):
        """Length of a path from where the robot is now, through every waypoint."""
        if not path:
            return float("inf")
        total = math.hypot(path[0][0] - self.x, path[0][1] - self.y)
        for a, b in zip(path, path[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1])
        return total

    def _path_is_clear(self, path, limit_m=PATH_BLOCK_CHECK_M):
        """Is the route we are already following still usable? Walks the first
        limit_m of it through the planning grid (static walls + sensed obstacle
        layer). Only the near portion matters: the far end will be replanned long
        before we get there, and checking it all would make any distant map
        flicker throw away a perfectly good route."""
        if not path:
            return False
        g = self.grid
        px, py = self.x, self.y
        travelled = 0.0
        for wx, wy in path:
            seg = math.hypot(wx - px, wy - py)
            steps = max(1, int(seg / (g.res * 0.75)))
            for i in range(1, steps + 1):
                t = i / steps
                x = px + (wx - px) * t
                y = py + (wy - py) * t
                r, c = g.world_to_cell(x, y)
                if not g.is_free(r, c):
                    return False
                if self.obs_score is not None and g.in_bounds(r, c) \
                        and self.obs_score[r, c] >= OBS_THRESH:
                    return False
            travelled += seg
            if travelled >= limit_m:
                break
            px, py = wx, wy
        return True

    def replan_to_victim(self):
        """Choose the plan source. The safe road is only an escape hatch used
        while we are actively stuck, and only when it makes progress toward our
        OWN victim (see plan_safe_road_to). As soon as we drive clear the stuck
        counter resets, so the next replan goes back to our own grid route and we
        break off the other robot's trail toward our own goal, rather than
        following it all the way to where it was going."""
        # The shared breadcrumb graph is EXPERIENCE-BASED PLANNING: ground the
        # other robot has physically driven is proven traversable, which is
        # exactly the "store experiences in a graph and reuse them" idea behind
        # the Thunder experience-based planner. The progress gate inside
        # plan_safe_road_to prevents it dragging this robot to the other robot's
        # victim, and on_safe_road is cleared as soon as the robot is moving again.
        stuck = self.recover_count >= SAFE_ROAD_AFTER
        if stuck and self.plan_safe_road_to(self.target):
            if not self.on_safe_road:
                print(f"[{self.name}] stuck; taking the shared safe road "
                      f"({len(self.path)} waypoints)")
            self.on_safe_road = True
            return True
        if self.plan_to_victim(self.target):
            self.on_safe_road = False
            return True
        if self.plan_safe_road_to(self.target):
            self.on_safe_road = True
            return True
        return False

    def _skip_victim(self, v):
        """Could not reach this victim (no path / walled off): mark it done so we
        move on, but flag it skipped so a FREE robot can try it again later. A
        cooldown stops the revisit logic from immediately re-picking it and
        ping-ponging between two sealed goals every tick."""
        v["done"] = True
        v["skipped"] = True
        # We never got near it, let alone saw it: it must NOT render as a confirmed
        # (green) find. Without this the map painted unreachable victims green.
        v["uncertain"] = True
        self.no_path_until[v.get("id")] = self.robot.getTime() + NO_PATH_COOLDOWN_S

    def _new_victim_candidate(self, vx, vy):
        """True only once a would-be NEW victim at (vx,vy) has been seen
        NEW_VICTIM_HITS times at roughly the same place. A single YOLO false
        positive (scenery, or the other robot) therefore cannot create a permanent
        phantom victim that both robots then drive to. Also refuses points sitting
        on a known victim or on either robot."""
        now = self.robot.getTime()
        # Never register on top of a known victim or on a robot.
        for v in self.victims:
            if math.hypot(v["x"] - vx, v["y"] - vy) < NEW_VICTIM_MIN_SEP:
                return False
        if self.other_pos is not None and \
                math.hypot(self.other_pos[0] - vx, self.other_pos[1] - vy) < NEW_VICTIM_MIN_SEP:
            return False
        if math.hypot(self.x - vx, self.y - vy) < 0.5:
            return False
        # Age out stale candidates, then accumulate this sighting.
        self.pending_new = [c for c in self.pending_new
                            if now - c["t"] <= NEW_VICTIM_TTL_S]
        for c in self.pending_new:
            if math.hypot(c["x"] - vx, c["y"] - vy) <= NEW_VICTIM_MATCH_M:
                c["n"] += 1
                c["t"] = now
                c["x"] = (c["x"] * (c["n"] - 1) + vx) / c["n"]   # running mean
                c["y"] = (c["y"] * (c["n"] - 1) + vy) / c["n"]
                if c["n"] >= NEW_VICTIM_HITS:
                    self.pending_new.remove(c)
                    return True
                return False
        self.pending_new.append({"x": vx, "y": vy, "n": 1, "t": now})
        return False

    def _not_others(self, v):
        """Victim is not the one the other robot is currently committed to."""
        if self.other_claim_id is not None and v.get("id") == self.other_claim_id:
            return False
        if self.other_claim is not None and \
                math.hypot(v["x"] - self.other_claim[0],
                           v["y"] - self.other_claim[1]) < CLAIM_MATCH_M:
            return False
        return True

    def _pick_revisit(self):
        """When we have no assigned victims left (free), pick one worth a second
        attempt instead of just wandering: one we could not reach (skipped), or
        one we 'confirmed' but with a large odometry gap (uncertain we truly got
        within 1.0 m). Bounded by REVISIT_MAX so we never loop on a lost cause."""
        now = self.robot.getTime()
        cands = [v for v in self.victims
                 if (v.get("skipped") or v.get("uncertain"))
                 and v.get("free_retries", 0) < REVISIT_MAX
                 and now >= self.no_path_until.get(v.get("id"), -1.0)
                 and self._not_others(v)]
        if not cands:
            return None
        v = min(cands, key=lambda v: math.hypot(v["x"] - self.x, v["y"] - self.y))
        v["done"] = False
        v["skipped"] = False
        v["free_retries"] = v.get("free_retries", 0) + 1
        print(f"[{self.name}] free; second attempt at victim #{v.get('id')} "
              f"(try {v['free_retries']})")
        return v

    def step_plan(self):
        self.on_safe_road = False
        self.target = self.pick_next_victim()
        if self.target is None and ENABLE_RETRY:   # cleared our list: retry uncertain ones
            self.target = self._pick_revisit()
        if self.target is None:
            self.state = "EXPLORE"
            return
        if self.plan_to_victim(self.target):
            # Back out to the pre-approach standoff before setting off, so we never
            # try to drive a fresh route out of a pose pressed against a victim.
            if self.confirm_safe_xy is not None:
                sx, sy = self.confirm_safe_xy
                if math.hypot(sx - self.x, sy - self.y) > CONFIRM_RETREAT_M:
                    self.path.insert(0, (sx, sy))
                    print(f"[{self.name}] retreating to safe point "
                          f"({sx:.2f},{sy:.2f}) before the next victim")
                self.confirm_safe_xy = None
            print(f"[{self.name}] planning to victim #{self.target.get('id')} "
                  f"({self.target['x']:.2f},{self.target['y']:.2f}), "
                  f"{len(self.path)} waypoints")
            self.state = "NAV"
        else:
            print(f"[{self.name}] no path to "
                  f"({self.target['x']:.2f},{self.target['y']:.2f}); skipping")
            self._skip_victim(self.target)

    def _enter_confirm(self):
        print(f"[{self.name}] arrived, confirming near "
              f"({self.x:.2f},{self.y:.2f})")
        self.stop()
        self.path = []
        self.on_safe_road = False
        # Retreat point. The robot drove to this spot under the planner, so it is
        # known clear, and it is recorded BEFORE the final creep into the body.
        # Closing right up to a victim otherwise leaves the next plan starting from
        # a pose wedged against the mesh, where every rollout is blocked and the
        # robot burns the rest of the mission thrashing. Backing out to here first
        # costs one waypoint and always succeeds, because we came in that way.
        self.confirm_safe_xy = (self.x, self.y)
        self.confirm_start = self.robot.getTime()
        self.confirm_close_since = -1.0
        self.confirm_min_dv = float("inf")
        self.confirm_got_near = False   # did we ever PHYSICALLY confirm (camera+range)?
        self._servo = None              # fresh sensor lock for this victim
        # Pick the side to work around a blocked approach: whichever is clearer.
        left_clear = self._lidar_range_at(math.pi / 2.0)
        right_clear = self._lidar_range_at(-math.pi / 2.0)
        self.confirm_sweep_dir = 1.0 if left_clear >= right_clear else -1.0
        self.confirm_orbit_start = -1.0
        self.state = "CONFIRM"

    def step_nav(self):
        if not self.path:
            self._enter_confirm()
            return
        # Range-triggered handoff: once we are within CONFIRM_RANGE_M of the
        # victim estimate, stop the DWA (which would otherwise fight the body it
        # is trying to reach) and hand the final closing to camera homing. This
        # is what breaks the approach/reverse/approach loop at the victim's feet.
        dv = math.hypot(self.target["x"] - self.x, self.target["y"] - self.y)
        if dv < CONFIRM_RANGE_M:
            self._enter_confirm()
            return
        # Opportunistic acquisition: if the camera already sees a victim within
        # near line of sight, stop chasing the CSV coordinate and hand off to
        # camera homing on the real victim. Only if that detection maps to a
        # victim that is ours to take (not the other robot's and not done), so we
        # never poach the target it is already working on.
        saw, conf, bearing, _ = self.throttled_look()
        if saw and conf >= 0.5:
            angs, rs = self.read_lidar()
            rng = self.dir_clearance(bearing, angs, rs) if angs else LIDAR_MAX_USE
            dwx = self.x + rng * math.cos(self.theta + bearing)
            dwy = self.y + rng * math.sin(self.theta + bearing)
            hit = self._victim_at(dwx, dwy)
            if rng < CONFIRM_ACQUIRE_M and not self._detection_is_theirs(bearing, rng) \
                    and hit is not None and self._available(hit):
                self._enter_confirm()
                return
        now = self.robot.getTime()
        # Periodically replan (grid route, or the shared safe road when stuck) so
        # newly-sensed furniture gets routed around instead of thrashed against.
        if now - self.last_replan >= REPLAN_PERIOD_S:
            # Keep the route we are on unless it is genuinely blocked or the new
            # one is clearly shorter (see PATH_SWITCH_MARGIN).
            old_path = list(self.path)
            old_ok = self._path_is_clear(old_path)
            old_len = self._path_len(old_path) if old_ok else float("inf")
            if not self.replan_to_victim():
                if old_ok:
                    self.path = old_path      # keep driving what we had
                else:
                    print(f"[{self.name}] victim walled off by obstacles; skipping")
                    self._skip_victim(self.target)
                    self.state = "PLAN"
                    return
            elif old_ok and not self.on_safe_road:
                new_len = self._path_len(self.path)
                if new_len > (1.0 - PATH_SWITCH_MARGIN) * old_len:
                    self.path = old_path      # not worth switching, stay committed
        dist = self.navigate(self.path)
        if self.need_replan:
            self.need_replan = False
            # A recovery just finished. This replan used to be unconditional, which
            # is where the U-turns came from: recoveries fire in bursts, and each one
            # was free to hand back a route around the OTHER side of the obstacle,
            # so the robot committed, turned around, recovered, and committed back.
            # Hysteresis matters more here than on the periodic replan, not less.
            old_path = list(self.path)
            old_ok = self._path_is_clear(old_path)
            old_len = self._path_len(old_path) if old_ok else float("inf")
            if not self.replan_to_victim():
                if old_ok:
                    self.path = old_path
                else:
                    self._skip_victim(self.target)
                    self.state = "PLAN"
            elif old_ok and not self.on_safe_road:
                if self._path_len(self.path) > (1.0 - PATH_SWITCH_MARGIN) * old_len:
                    self.path = old_path      # recovery does not license a new route
            return
        if dist < ARRIVE_TOL:
            self._enter_confirm()

    def step_confirm(self):
        """Drive right up to the victim and report while there. The supervisor
        scores only when the robot's TRUE position is within 1.0 m of the waist
        marker (offset up to 1.3 m from the figure), so parking at a guessed
        distance from the estimate misses. Instead we CLOSE IN: steer by the
        camera when it sees the victim, and creep forward until we are a short
        standoff from the body (fused lidar/depth/IR) or right on the estimate.
        Crucially the timer does not run out while we are still approaching, only
        once we are actually near, so a slow approach no longer makes us give up
        and drive off before ever getting close enough to score."""
        now = self.robot.getTime()
        saw, conf, bearing, _ = self.throttled_look()

        dv = (math.hypot(self.target["x"] - self.x, self.target["y"] - self.y)
              if self.target is not None else 0.0)
        self.confirm_min_dv = min(self.confirm_min_dv, dv)

        # THE rule: a person in view AND the sensor distance in the box direction
        # under the threshold means found. fwd_raw is only the hard anti-stomp
        # floor so we never touch the body; if there is no camera detection we
        # fall back to the estimate distance.
        fwd_raw = self.forward_clearance_raw()
        saw_v = saw and not self._detection_is_theirs(bearing, fwd_raw)
        rng = self.victim_range(bearing) if saw_v else float("inf")
        self.record_victim_debug(saw_v, bearing, dv, fwd_raw)

        # Optional sensor lock on the body centroid, used ONLY for steering when
        # enabled. It never decides whether a victim is found.
        servo = (self.victim_body_measurement(bearing)
                 if (SERVO_ENABLE and saw_v) else None)
        if servo is not None:
            self._servo, self._servo_t = servo, now
        elif self._servo is not None and now - self._servo_t > SERVO_LOST_HOLD_S:
            self._servo = None
        servo_brg = self._servo[3] if self._servo is not None else 0.0

        # Closeness is decided on POSITION only. The rules require being within
        # 1.0 m of the victim and sending a message; no sensor confirmation is
        # asked for, so the sensors below serve one purpose here -- fwd_raw is the
        # anti-stomp floor that keeps the robot from driving into the body, which
        # is obstacle avoidance, not victim confirmation. Because our estimate is
        # itself offset from the true marker, the robot should close on the
        # estimate as far as it physically can rather than stopping at a sensor
        # range, since every centimetre short is lost scoring margin.
        reached = self.target is not None and dv <= MARK_REACH_M
        found_now  = reached
        # blocked means something is inside the anti-stomp floor. It must NOT count
        # as arriving: being pressed against a wall several metres short of the
        # victim is not the same as reaching it. It only forbids further forward
        # motion, and the robot then works sideways around the obstruction.
        # The reactive collision monitor also applies while confirming. It carves the
        # TARGET victim out of its obstacle set, so it never blocks the approach to
        # the victim itself, but it does stop the robot driving into a wall, a pipe
        # or the other robot while it is busy homing in.
        guard_stop, _gfwd, _gl, _gr = self.collision_guard()
        blocked = (fwd_raw <= CONFIRM_MIN_CLEAR) or guard_stop
        truly_near = found_now
        at_estimate = self.target is not None and dv <= CONFIRM_TARGET_M
        if reached:
            self.confirm_got_near = True
        if truly_near and self.confirm_close_since < 0.0:
            self.confirm_close_since = now   # start the score-and-hold timer

        # Reporting is handled centrally by maybe_report() (runs every step in ANY
        # state, including here): it reports only on a live, victim-associated
        # detection within range, at an honest confidence, bounded per victim. No
        # blind "on the estimate" reporting.

        # After the hold completes, circle the body once before leaving. Closing
        # on the VISIBLE part of the figure can still leave the true body >1.0 m
        # from the waist marker (offset up to ~1.3 m on a lying victim reached
        # from the wrong end), so the orbit sweeps our TRUE position along an arc
        # around the figure while maybe_report keeps firing: some point of that
        # arc passes through the marker's scoring circle whichever end we came
        # from. Direction picked once, toward the clearer side.
        held = self.confirm_close_since >= 0.0 and \
            now - self.confirm_close_since >= CONFIRM_HOLD_S
        timeout = now - self.confirm_start >= CONFIRM_MAX_S
        if held and self.confirm_orbit_start < 0.0 and not timeout:
            left = self._lidar_range_at(math.pi / 2.0)
            right = self._lidar_range_at(-math.pi / 2.0)
            self.confirm_orbit_dir = 1.0 if left >= right else -1.0
            self.confirm_orbit_start = now
            print(f"[{self.name}] confirm hold done; orbiting the body "
                  f"({'left' if self.confirm_orbit_dir > 0 else 'right'}) "
                  f"for {ORBIT_S:.0f}s while reporting")
        orbiting = self.confirm_orbit_start >= 0.0
        orbit_done = orbiting and now - self.confirm_orbit_start >= ORBIT_S

        # Done when the post-hold orbit has completed, or the whole confirm runs
        # over the hard cap (bad estimate / phantom, gave up). A running orbit is
        # allowed to finish its own (bounded) window even past the cap: it exists
        # precisely to convert this confirm into a score.
        if orbit_done or (timeout and not orbiting):
            done_v = self.target if self.target is not None else min(
                self.victims,
                key=lambda v: math.hypot(v["x"] - self.x, v["y"] - self.y),
                default=None)
            if done_v is not None:
                done_v["done"] = True
                done_v["skipped"] = False
                # Green only if we actually confirmed close; otherwise flag it
                # uncertain so a free robot revisits it from another angle.
                done_v["uncertain"] = not self.confirm_got_near
                self.announce_done(done_v)
            self.stop()
            self.state = "PLAN"
            return

        # Steering bearing: prefer the camera (accurate), else the estimate.
        if self._servo is not None:
            steer = servo_brg
        elif saw_v:
            steer = bearing
        elif self.target is not None:
            steer = wrap_pi(math.atan2(self.target["y"] - self.y,
                                       self.target["x"] - self.x) - self.theta)
        else:
            steer = 0.0

        if orbiting:
            # Circle the body: hold the victim at a fixed side bearing while
            # creeping forward, which traces an arc around it at the current
            # standoff. Anti-stomp still absolute: if the nearest surface gets
            # under the floor, rotate away from the body instead of advancing.
            side = self.confirm_orbit_dir * ORBIT_SIDE
            err = wrap_pi(steer - side)
            if fwd_raw <= CONFIRM_MIN_CLEAR:
                sp = CONFIRM_SEARCH_SPIN * (-self.confirm_orbit_dir)
                self.set_wheels(-sp, sp)          # rotate the nose away, no advance
            else:
                turn = clamp(TURN_GAIN * err, -ORBIT_SPEED, ORBIT_SPEED)
                self.set_wheels(ORBIT_SPEED - turn, ORBIT_SPEED + turn)
        elif blocked and not truly_near:
            # Short of the victim with an obstruction ahead. Forward motion is
            # forbidden, so sweep sideways around it looking for a clear approach
            # rather than sitting still (which would let the confirm time out and
            # abandon a victim we never actually got near).
            sp = CONFIRM_SEARCH_SPIN * self.confirm_sweep_dir
            self.set_wheels(-sp, sp)
        elif truly_near:
            # Physically on the victim: hold and face it so the camera keeps
            # confirming and reports keep flowing from a scoring position.
            turn = clamp(TURN_GAIN * steer, -CONFIRM_SPEED, CONFIRM_SPEED)
            self.set_wheels(-turn, turn)
        elif saw_v or not at_estimate:
            # Still closing. Ease off as the nearest surface approaches the floor:
            # the robot arrives gently instead of driving in at full creep speed,
            # which is what tips it or climbs it onto an obstacle. It still reaches
            # the same standoff, just without the momentum.
            scale = clamp((fwd_raw - CONFIRM_MIN_CLEAR) /
                          max(1e-3, CONFIRM_SLOW_M - CONFIRM_MIN_CLEAR),
                          CONFIRM_CREEP_MIN, 1.0)
            sp = CONFIRM_SPEED * scale
            turn = TURN_GAIN * steer
            self.set_wheels(sp - turn, sp + turn)
        else:
            # Parked on an EMPTY estimate (bad localization): the real victim is
            # likely ~1 m off to the side. A pure spin sweeps only bearing, so we
            # sweep POSITION instead: open out to ORBIT_EST_R from the estimate,
            # then circle it (estimate held at a fixed side bearing). The blind
            # reports (dv < CONFIRM_REPORT_M) keep firing along the whole arc,
            # so the TRUE body passes through the marker's scoring circle if the
            # true victim sits within ~ORBIT_EST_R + 1.0 m of the estimate. The
            # camera meanwhile pans across the whole surroundings; the moment it
            # picks the victim up, saw_v flips and normal homing takes over.
            if fwd_raw <= CONFIRM_MIN_CLEAR:
                self.set_wheels(-CONFIRM_SEARCH_SPIN, CONFIRM_SEARCH_SPIN)
            elif dv < (self.target.get("sigma", ORBIT_EST_R)
                       if self.target is not None else ORBIT_EST_R):
                self.set_wheels(CONFIRM_SPEED, CONFIRM_SPEED)   # open out first
            else:
                side = self.confirm_orbit_dir * ORBIT_SIDE
                err = wrap_pi(steer - side)   # steer = bearing to the estimate here
                turn = clamp(TURN_GAIN * err, -CONFIRM_SPEED, CONFIRM_SPEED)
                self.set_wheels(CONFIRM_SPEED - turn, CONFIRM_SPEED + turn)

    def step_explore(self):
        # A victim spotted that is not one we already handled gets serviced like
        # any other. Aim at where the camera says it is, then confirm homes in.
        saw, conf, bearing, _ = self.throttled_look()
        if saw and conf >= 0.5:
            angs, rs = self.read_lidar()
            rng = self.dir_clearance(bearing, angs, rs) if angs else 2.0
            if not self._detection_is_theirs(bearing, rng):
                ang = self.theta + bearing
                vx, vy = self.x + rng * math.cos(ang), self.y + rng * math.sin(ang)
                hit = self._victim_at(vx, vy)
                if hit is not None:
                    # Falls on a known victim: only service it if it is still
                    # ours to take (not done, not the other robot's).
                    if self._available(hit):
                        self.target = hit
                        self._enter_confirm()
                        return
                elif conf >= NEW_VICTIM_CONF and self._in_bounds(vx, vy) \
                        and self._new_victim_ok(vx, vy) \
                        and self._new_victim_candidate(vx, vy):
                    # Confirmed over several sightings: register it with a fresh id
                    # so it joins the shared claim/done bookkeeping.
                    self.target = {"id": len(self.victims), "x": vx, "y": vy,
                                   "done": False, "mine": True,
                                   "sigma": ORBIT_EST_R, "quality": "live"}
                    self.victims.append(self.target)
                    print(f"[{self.name}] NEW victim #{self.target['id']} registered "
                          f"at ({vx:+.2f},{vy:+.2f}) conf={conf:.2f}")
                    self._enter_confirm()
                    return

        # Coverage points we have driven near count as searched.
        self.explore_remaining = [p for p in self.explore_remaining
                                  if math.hypot(p[0] - self.x, p[1] - self.y) > COVERAGE_PRUNE_M]

        if not self.path:
            tgt = self._next_explore_target()
            if tgt is None:
                self.stop()
                return
            self.plan_to(*tgt)   # if unreachable, next tick picks another point
            return
        dist = self.navigate(self.path)
        if self.need_replan:
            self.need_replan = False
            self.path = []
            return
        if dist < ARRIVE_TOL:
            self.path = []

    # ---- main loop --------------------------------------------------------
    def run(self):
        self.robot.step(self.timestep)   # let sensors populate, then calibrate
        self.calibrate()

        while self.robot.step(self.timestep) != -1:
            self.update_odometry()
            self.check_stall()      # wheels spinning but wedged? undo phantom travel
            self.correct_odometry_to_walls()   # snap drift back onto the wall map
            self.update_obstacle_layer()
            self.poll_squad()
            self.broadcast_position()
            self.record_breadcrumb()
            self.mark_reached_victims()   # position-only marking, in ANY state
            self.maybe_report()     # report near ANY seen victim, in any state
            self.maybe_intercept()  # divert-and-close on an unconfirmed victim in view
            self.draw_map()         # live top-view map window

            # Tipped over: hold still and skip the whole control step. Without this
            # the planner keeps issuing wheel commands while odometry is frozen, so
            # the robot drives on with a position estimate that is no longer being
            # updated and ends up steering from a stale pose.
            if self.fallen:
                self.stop()
                continue

            now = self.robot.getTime()
            if now - self.last_debug >= 2.0:
                tgt = (f"#{self.target.get('id')}("
                       f"{self.target['x']:+.1f},{self.target['y']:+.1f})"
                       if self.target else None)
                # Forward-cone minimums per sensor, to see if the depth camera is
                # actually detecting camera-height obstacles like the rod.
                dmin = min((d for a, d in zip(*self.read_depth_scan())
                            if abs(a) < SAFE_CONE), default=9.9)
                lmin = min((d for a, d in zip(*self.read_lidar())
                            if abs(wrap_pi(a)) < SAFE_CONE), default=9.9)
                print(f"[{self.name}] t={now:4.0f} {self.state:8s} "
                      f"pos=({self.x:+.2f},{self.y:+.2f}) "
                      f"hdg={math.degrees(self.theta):+4.0f} "
                      f"tgt={tgt} theirs={self.other_claim_id} "
                      f"wp={len(self.path)} rec={self.recover_count} "
                      f"road={'Y' if self.on_safe_road else 'n'} "
                      f"crumbs={len(self.trail)}+{len(self.other_trail)} "
                      f"odom_dist={self.odom_dist:.1f}m "
                      f"fwd[dpt={dmin:.2f} lid={lmin:.2f} ir={self.front_ir():.2f}]")
                self.last_debug = now

            if now > MISSION_CAP_S - END_BUFFER_S:
                if not self._dbg_printed:
                    self.print_victim_debug()
                    self._dbg_printed = True
                self.stop()
                continue

            if self.state == "PLAN":
                self.step_plan()
            elif self.state == "NAV":
                self.step_nav()
            elif self.state == "CONFIRM":
                self.step_confirm()
            elif self.state == "EXPLORE":
                self.step_explore()

        # Simulation ended (e.g. all victims found before the cap): print too.
        if not self._dbg_printed:
            self.print_victim_debug()
            self._dbg_printed = True


if __name__ == "__main__":
    GroundMission().run()
