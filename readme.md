# IEEE SMCS Search and Rescue Competition 2026 — Phase 1 Submission

Two-part solution for the Phase 1 SAR task:

1. **Flyover information extraction** (`src/sar_pipeline.py`) — processes the
   pre-recorded UAV footage into victim position estimates and a wall map.
2. **Ground fleet control** (`proposed_solution.py`) — the Webots controller that
   drives both ROSbots to those victims, confirms them and reports them.

---

## 1. Environment

Python **3.11** (any 3.10+ works).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Then point Webots at that interpreter: `Tools > Preferences > General > Python command`.

Install **only** `opencv-contrib-python` (already pinned in `requirements.txt`).
Having both it and `opencv-python` in one environment can shadow `cv2.aruco`,
which the pipeline needs to establish the origin-marker reference frame.

---

## 2. Running a mission

### Step 1 — process the flyover footage

```bash
python src/sar_pipeline.py
```

The world is selected by a single line at the top of `src/sar_pipeline.py`:

```python
VIDEO_PATH = os.path.join(_COMP_ROOT, "recordings", "large_world_flyover.mp4")
```

Everything else (IMU csv, cache folder, output folder) is derived from that name.

- **First run for a world** performs the full video pass (several minutes) and
  writes the deliverables plus a reusable cache.
- **Every later run** for the same world takes about a second: it just copies that
  world's deliverables into `sim_logs/` and exits.

### Step 2 — run the simulation

Open the matching world in Webots (`worlds/large_world.wbt`) and press play.

> **The world processed in step 1 must match the `.wbt` you open.** The
> deliverables in `sim_logs/` belong to one world at a time. If they disagree, the
> controller prints a `WORLD MISMATCH` banner in the first second of the run
> instead of silently driving to another world's coordinates.

### Optional — verify the estimates without Webots

```bash
python src/evaluate_estimates.py               # the currently active world
python src/evaluate_estimates.py small_world   # any processed world by name
```

Parses the true victim positions out of the `.wbt` and the marker offsets out of
`protos/Victim.proto`, then scores our submitted CSV exactly the way the marking
supervisor does.

---

## 3. Layout

```
proposed_solution.py     Webots controller for both ROSbots
requirements.txt
readme.md
src/
  sar_pipeline.py        flyover pipeline (odometry -> walls -> victims)
  evaluate_estimates.py  offline accuracy check against ground truth
  yolov8n.pt             person detector, ground robot cameras
  yolo11m.pt             person detector, flyover footage
  best_segmentation.pt   wall/floor segmentation, flyover footage
  <world>/mapping_data/  reusable per-world cache (regenerable)
sim_logs/
  victim_location_estimates.csv   scored deliverable, active world
  map_estimate.png                scored deliverable, active world
  wall_estimates.csv              vector wall map read by the controller
  <world>/                        per-world archive of the above
```

`sim_logs/` holds the **active** world at its root, because the marking supervisor
reads those exact paths, and a per-world archive in subfolders so switching worlds
never destroys previous results.

The `.pt` files are model weights (data, not compiled objects). They are bundled
because the rules require solutions to be self-contained with no external calls,
which rules out downloading weights at runtime.

---

## 4. How it works

### Flyover pipeline

1. **Visual odometry** — recovers the UAV trajectory from the footage. Scale and
   the reference frame come from the origin-marker ArUco patterns; altitude is
   derived from the known marker size.
2. **Wall mapping** — a segmentation model masks walls per frame; masks are
   stamped into a global occupancy grid using the recovered pose, then vectorised
   into line segments and rasterised to the required 600×600 @ 0.05 m/px map.
3. **Victim localisation** — a person detector runs on every tracked frame. Each
   detection is projected to the ground plane and clustered across frames. A
   cluster's position is solved by **Huber-robust multi-view ray triangulation**
   (least-squares closest point to all viewing rays) rather than averaging the
   ground projections, because averaging leaves a systematic
   `body_height × tan(off-nadir)` parallax bias that no amount of averaging
   removes. A nadir-weighted geometric median is the fallback when ray geometry is
   degenerate.

Outputs are written in the **origin-marker reference frame**, which is what the
marking pipeline expects.

### Ground controller

Both robots run from one controller, distinguished by name.

- **Odometry** — wheel encoders for distance, compass for absolute heading (so
  heading cannot drift). Accurate to a few centimetres over a full mission.
- **Task allocation** — victims are split by a greedy route-cost balance so both
  robots start on a nearby victim and carry comparable travel, which serves both
  the coordination and efficiency criteria.
- **Global planning** — A* over a layered costmap (static walls from the flyover
  map, plus a live log-odds obstacle layer from lidar and depth). Routes are
  committed rather than replaced on every replan, so a flickering costmap cell
  cannot flip the robot between two equal-cost ways around an obstacle.
- **Local planning** — Dynamic Window Approach for velocity selection, with
  Follow-the-Gap supplying the heading in clutter. Gap geometry is read directly
  off the scan in one pass, which avoids the stall that pure trajectory sampling
  hits in narrow openings.
- **Safety** — a reactive collision monitor below the planner fuses lidar, the
  depth camera (for obstacles above or below the lidar plane) and the IR ring.
  Wheel-slip and tip-over are detected from the scan and the accelerometer, and
  suspend odometry integration rather than letting a bad estimate propagate.
- **Confirmation** — on arrival the robot closes on the victim, holds while
  reporting, then orbits the estimate. Because the estimate is offset from the
  true waist marker by an unknown direction, the orbit sweeps the robot's true
  distance through the scoring radius from whichever side it approached.
- **Reporting** — `victim_found` is sent on odometry proximity to an unfound
  victim, with the camera raising the reported confidence when it also sees a
  person. Sends are bounded per victim.

### Shared experience map

Both robots broadcast breadcrumbs of where they have actually driven. Since driven
ground is proven traversable, a robot whose grid route keeps failing can plan over
the other robot's breadcrumb graph instead. This is the experience-based planning
idea used by the Thunder planner, restricted so that it only ever helps progress
toward the robot's own victim.

---

## 5. Configuration

Pipeline (`src/sar_pipeline.py`):

| Setting | Purpose |
|---|---|
| `VIDEO_PATH` | selects the world; everything else derives from it |
| `FORCE_ODOMETRY` | redo the video pass even if a cache exists |
| `VICTIM_CONF` | detector confidence floor for flyover victims |
| `MIN_DETECTION_HITS` | frames a cluster needs before it counts as a victim |

Controller (`proposed_solution.py`):

| Setting | Purpose |
|---|---|
| `MARK_REACH_M` | odometry distance at which a victim counts as reached |
| `FGM_ENABLE` | Follow-the-Gap heading selection on top of DWA |
| `DEBUG_MAP` | live top-view map window (off by default) |
| `MISSION_CAP_S` | mission time budget, 180 s |

Per-run console output from both robots is also written to
`run_logs/<timestamp>_<world>/`.

---

## 6. Known limitations

- **Victim estimate accuracy varies by world.** On `large_world` the mean error is
  about 1.0 m; on `medium_world` it is about 2.6 m, dominated by drift along the
  UAV's main axis of travel that grows with distance flown. Since scoring requires
  the robot to be within 1.0 m of the true marker, this is the single largest
  limit on victim-finding performance, and it is upstream of the controller.
- **Marker vs body.** Scoring is measured to the victim's waist marker, which the
  PROTO offsets up to 1.3 m from the mesh origin. The pipeline estimates the
  visible body, so the post-arrival orbit exists to cover that offset.
- `FORCE_ODOMETRY = True` rewrites a world's deliverables as it runs, so an
  interrupted forced run leaves that world's estimates incomplete until it is
  re-run to completion.
