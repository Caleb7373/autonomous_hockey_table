# Air hockey: four motors, four strings, striker and puck

This is a Python desktop simulation for Caleb's cable-driven air hockey project.
It animates the striker striking a free puck, draws four strings from motor/reel
exit points to the striker hub, and records the actual tension in each string at
every integration step. Default recording interval: **0.0001 s (0.1 ms)**.

The strings attach to the **striker**, not to the puck. The displayed motor
locations are also the ideal cable exit points. Front means toward the opponent
(top of the drawing). M1/M2 are rear; M3/M4 are front.

## Start in VS Code (Windows)

1. Extract the whole ZIP. Keep `air_hockey_sim.py` and `physics.py` together.
2. In VS Code, choose **File > Open Folder** and open the extracted folder
   containing those two files. Select your Python interpreter if needed.
3. Open **Terminal > New Terminal** and run:

```bash
python -m pip install -r requirements.txt
python air_hockey_sim.py
```

Python 3.10 or newer is recommended. Standard Windows Python includes Tkinter.
If `python` is unavailable but the Windows Python launcher is installed, use
`py` in place of `python` in both commands. On Linux, Tkinter may require your
system's `python3-tk` package. On a remote/headless computer use the command below.

The first run takes several seconds to calculate. Playback then starts. Changes
to the inputs apply when you click **Simulate**, not to the already computed run.

## Controls

- **Motion**: table dimensions, anchor inset, start/target positions, masses,
  radii, desired speed/acceleration, drag, and initial puck velocity.
- **Strings**: cable stiffness/damping, target pretension bounds, puck contact
  stiffness/damping, and wall contact stiffness/damping.
- **Control**: position, speed and tension feedback gains; force allocation.
- **Time**: run duration, integration timestep, controller update period and
  friction smoothing. Control period must be an integer multiple of timestep.
- **Motors**: M1–M4 can each use a different type and parameters. You can put
  larger motors at the front and smaller ones at the rear. **Copy M1 settings
  to all four motors** makes uniform comparisons easy.
- **Play/Pause, Rewind, playback speed**: slow down the strike without changing
  the physics. Drag the time slider to inspect any recorded instant.
- **Live calculation**: select a motor to see geometry, string extension,
  spring/damping contributions, calculated tension, torque and acceleration
  with the actual numeric substitutions at the selected time.
- **Equations**: all model equations and conventions.
- **Run summary**: peak loads, speeds, contact, tracking error, slack duration,
  time at torque limits and time above the continuous-torque reporting threshold.
- **Export CSV**: all four tensions, all motor speeds and torques, string lengths,
  contact force, striker/puck position and velocity at every integration step.
- **Save/Load settings**: preserve and compare experiments as JSON.

Dashed translucent strings indicate zero tension (slack). They remain drawn as
straight geometric connections; their sagging shape is not modeled. Reel spokes
rotate to show winding direction. The dashed path and cross indicate the target
trajectory. Colored tension dots and a black time cursor follow playback. Plots
show the whole computed run to make peaks easy to find. Their display is reduced
to about 5,000 samples for responsiveness; **CSV export retains all samples**.

## What to try first

1. Run the default case; watch contact near **0.367 s**. The assumed setup hits
   the puck, which subsequently enters the opponent's goal.
2. Open **Live calculation**, select M3 and move the time slider around impact.
3. In **Motors**, reduce `peak torque Nm` on each motor. Also lower `continuous
   torque Nm` if necessary so it does not exceed peak. Run again and compare
   tracking error, tension, torque-limited time and whether contact occurs.
4. Change reel diameter while retaining the same motor. A smaller reel changes
   both load torque and required RPM/angular acceleration. Update reel inertia
   too if the physical reel changes; the simulation will not invent its mass.
5. Change `cable stiffness N m` or contact damping and inspect tension peaks.
   If you raise stiffness significantly, the program may ask for a smaller
   integration timestep. These parameters must ultimately be measured.

The default motor intentionally reaches its torque envelope during part of the
motion, and the rear strings briefly go slack. The simulator displays these
limitations rather than forcing perfect tracking or positive tension.

## Adjustable motor types

**All included profiles are illustrative, not verified commercial products.**
Selecting a type does not automatically identify a real motor. You must enter
its usable torque/speed information, inertia, reel size and driver response.
Every type is modeled as an ideal feedback actuator with limited torque.

| Type | Illustrative torque envelope |
|---|---|
| Stepper | `peak / (1 + (RPM/knee_RPM)^2)` |
| DC | `peak * max(0, 1 - RPM/no_load_RPM)` |
| Servo | Constant torque to base RPM, then constant power |
| Custom | Linear interpolation of your `rpm,torque_Nm` CSV |

The characteristic-speed field means knee RPM for stepper, no-load RPM for DC,
and base RPM for servo. Torque becomes zero above `maximum speed rpm`; this is
an envelope limit, not a physical speed clamp. External cable forces can still
backdrive a reel. The same symmetric envelope is assumed for braking and winding.
A real drive can behave differently in those operating regions.

For a real motor, choose **Load torque-speed CSV** and use a file like:

```csv
rpm,torque_Nm
0,0.9
300,0.7
600,0.4
900,0.2
1200,0.08
```

Those rows are an **invented format example**, not a manufacturer curve. The file
must start at 0 RPM and contain unique nonnegative speeds and nonnegative torques.
Outside the supplied range, custom torque is zero (no extrapolation). It is also
capped by the editable peak torque and maximum speed. Put the true usable limit
in those fields. Continuous torque is only a reporting threshold; exceeding it
is logged, not used as an automatic temperature or shutdown prediction.

Stepper holding torque is not torque at operating speed. Use a curve for the
actual driver supply and current. This simulation does NOT model step pulses,
microsteps, missed steps, rotor electrical angle or the actual closed-loop
algorithm of a particular kit. A DC or brushless motor similarly needs a real
driver and feedback system to implement this controller.

Primary background references (presets were not fitted to these documents):
- Oriental Motor, motor/driver speed–torque characteristics and holding torque:
  https://www.orientalmotor.com/products/pdfs/F_TecRef/TecRefAll.pdf
- maxon, motor data and simulation:
  https://support.maxongroup.com/hc/en-us/articles/360013761160-Motor-data-and-simulation

## Explicit physics

`physics.py` contains the calculation code. Its module docstring defines every
physical equation. Important functions are:

| Function | Purpose |
|---|---|
| `Model.geometry` | Cable lengths and directions |
| `Model.strings` | Actual tension from stretch and stretch rate |
| `Model.limits` | Each motor's torque at its current RPM |
| `Model.control` | Position/velocity feedback and tension/torque commands |
| `Model.derivative` | Coupled striker, puck, reel and motor acceleration |
| `contact` | Finite-duration, equal-and-opposite puck contact force |
| `simulate` | RK4 time integration and recording |
| `export_csv` | Full-resolution telemetry |

For each string:

```text
L = distance from anchor to striker
l = initial_unstretched_length - reel_radius * motor_angle
extension = L - l
extension_rate = -cable_direction dot striker_velocity + radius * motor_speed
T = max(0, stiffness*extension + damping*extension_rate) when extension > 0
T = 0 when extension <= 0
```

The controller's minimum and maximum tensions are **targets**. They are NOT
clamps on actual forces. Actual tension may fall to zero or exceed the target
maximum in a transient. Contact forces enter the striker and puck dynamics
with equal magnitudes and opposite directions. The motor responds through its
finite inertia, torque envelope and response time, instead of maintaining a
commanded speed perfectly during the collision.

Cable elasticity is an effective spring/damper; constant stiffness ignores its
variation with paid-out length. Contact uses a spring/damper with clipped
nonnegative normal force. The small visible overlap of the two disks represents
contact deformation. Restitution is not independently imposed: changing contact
stiffness/damping changes the resulting rebound.

Efficiency is approximated as a passive friction torque:

```text
loss = T*r*(1/eta - 1)*tanh(omega/omega_smoothing)
J*alpha = actual_motor_torque - T*r - viscous_friction*omega - loss
```

At positive winding speed this approaches the familiar steady torque `T*r/eta`.
This lumped model always dissipates mechanical energy through the loss term.
It does not compute electrical efficiency, current, bus voltage or regeneration.

## Scope and interpretation

- Fixed-radius, direct-drive reels; all axes have separate motor/reel inertia.
- Ideal nonrotating central striker attachment, no striker spin or tipping.
- No cable–puck, cable–cable or finite guide-pulley collisions. The drawing is
  schematic; a physical build needs cable routing that clears the puck.
- Uniform planar table; smoothed friction, no air-flow map or surface defects.
- Smooth motor torque model; ideal state feedback, no sensor noise/delay.
- Straight prescribed strike trajectory; no camera, AI or reactive puck tracking.
- No line breaking, thermal model, gear train, reel layering or mechanical stops.
- The open goal uses a simplified disk-clearance test; goal-post impacts are not
  resolved as separate rounded post collisions. The puck freezes once scored.
- Parameters are hypotheses. Predicted tension peaks are NOT certified hardware
  loads. Calibrate string stiffness/damping, contact behavior, motor data and drag.

## Run without a desktop window

```bash
python air_hockey_sim.py --headless --output results
python air_hockey_sim.py --config example_output/settings.json --headless --output rerun
```

This produces a full telemetry CSV, settings JSON, summary JSON and a PNG showing
the peak-contact frame. A precomputed example run is included in `example_output`.
The optional `example_output/playback.gif` shows that same run at slower speed.

## Numerical checks

```bash
python -m unittest -v test_physics.py
```

Seven tests check unilateral string tension, near-balanced initial pretension,
equal-and-opposite isolated contact and momentum conservation, dissipative loss,
speed-dependent torque limiting, cable geometry derivatives, default strike and
symmetry, timestep refinement, and sensitivity to weaker motors. These check
the implementation, not agreement with physical hardware.

The default timestep refinement compares 0.1 ms against 0.05 ms, including the
collision. Accepted tolerance is 3% for each peak cable tension and 0.004 in the
SI position/velocity state components at the end of that comparison run.
Different stiffnesses and gains can need a smaller timestep. The numerical
input guard is only a preliminary stability check, not a convergence proof.
