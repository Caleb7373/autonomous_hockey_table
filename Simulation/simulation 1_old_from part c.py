"""Part C: adjustable motor/reel sizing explorer (Python 3.10+).
Install: python -m pip install numpy matplotlib
Run:     python part_c_simulation.py
No GUI:  python part_c_simulation.py --headless

Edit CONFIG below or use the window. Click Recalculate after edits.
All efficiencies are fractions (0.90 = 90%). SI units unless labelled.

Model: direct drive, fixed effective winding radius, one winding-in cable.
This is a sizing/parameter simulation, not a closed-loop robot simulation.
Torque is a conservative sum of load and inertia magnitudes; it does not
model regenerative payout, impacts, cable elasticity, or controller response.
'center' models a symmetric rectangular layout at its center, goal along y.
The instantaneous geometry correction is included in center cable acceleration.
Motor/reel inertia stays fixed during diameter sweeps: update it for real reels.
No motor is declared suitable without an actual torque-speed curve.
"""
from dataclasses import dataclass, fields, replace
import argparse
import json
import math
from pathlib import Path


@dataclass
class Settings:
    mode: str = 'direct'  # 'direct' loads or 'center' values derived from Part B
    reel_diameter_mm: float = 90.0  # effective diameter at cable centerline
    mechanical_efficiency: float = 0.90  # cable power / shaft load power
    motor_driver_efficiency: float = 0.80  # shaft power / electrical power estimate
    rotor_inertia_kg_m2: float = 0.00010
    reel_inertia_kg_m2: float = 0.00003
    extra_shaft_friction_Nm: float = 0.0
    torque_margin_factor: float = 1.25  # sizing allowance, not actual power use
    direct_tension_N: float = 10.0
    direct_cable_speed_m_s: float = 2.0
    direct_cable_accel_m_s2: float = 10.0
    striker_mass_kg: float = 0.200
    striker_initial_speed_m_s: float = 0.0
    striker_target_speed_m_s: float = 2.0
    acceleration_distance_m: float = 0.200
    resistance_N: float = 1.0
    rear_tension_each_N: float = 4.0
    workspace_width_m: float = 0.720
    workspace_length_m: float = 0.720
    sweep_min_diameter_mm: float = 20.0
    sweep_max_diameter_mm: float = 120.0
    sweep_points: int = 200
    # Optional real manufacturer's data: CSV header rpm,torque_Nm.
    # Use the curve for your actual driver voltage/current. No extrapolation.
    motor_curve_csv: str = ''


CONFIG = Settings()  # You can edit any defaults above or set CONFIG fields here.


def validate(c):
    if c.mode not in ('direct', 'center'):
        raise ValueError('Mode must be direct or center.')
    for f in fields(c):
        value = getattr(c, f.name)
        if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
            raise ValueError(f'{f.name} must be finite and nonnegative.')
    for name in ('mechanical_efficiency', 'motor_driver_efficiency'):
        if not 0 < getattr(c, name) <= 1:
            raise ValueError(f'{name} must be greater than 0 and at most 1.')
    for name in ('reel_diameter_mm', 'striker_mass_kg', 'acceleration_distance_m',
                 'workspace_width_m', 'workspace_length_m', 'sweep_min_diameter_mm'):
        if getattr(c, name) <= 0:
            raise ValueError(f'{name} must be positive.')
    if c.torque_margin_factor < 1:
        raise ValueError('Torque margin factor must be at least 1.')
    if c.sweep_max_diameter_mm <= c.sweep_min_diameter_mm or not 2 <= c.sweep_points <= 10000:
        raise ValueError('Sweep needs increasing diameter limits and 2–10000 points.')
    if c.striker_target_speed_m_s < c.striker_initial_speed_m_s:
        raise ValueError('Center acceleration model requires target speed >= initial speed.')


def cable_loads(c):
    if c.mode == 'direct':
        return c.direct_tension_N, c.direct_cable_speed_m_s, c.direct_cable_accel_m_s2
    # Instantaneous forward cable at the center; symmetry cancels lateral forces.
    half_w, half_l = c.workspace_width_m / 2, c.workspace_length_m / 2
    length = math.hypot(half_w, half_l)
    cosine = half_l / length
    acceleration = (c.striker_target_speed_m_s**2 - c.striker_initial_speed_m_s**2) / (2*c.acceleration_distance_m)
    tension = c.rear_tension_each_N + (c.striker_mass_kg*acceleration + c.resistance_N)/(2*cosine)
    speed = c.striker_target_speed_m_s*cosine
    # L'' = -e.a + (v.v - (e.v)^2)/L; winding-in acceleration = -L''.
    winding_accel = acceleration*cosine - c.striker_target_speed_m_s**2*(1-cosine**2)/length
    return tension, speed, abs(winding_accel)


def calculate(c, diameter_mm=None):
    validate(c)
    diameter = c.reel_diameter_mm if diameter_mm is None else diameter_mm
    if diameter <= 0:
        raise ValueError('Diameter must be positive.')
    r = diameter/2000
    tension, speed, accel = cable_loads(c)
    omega = speed/r
    load = tension*r/c.mechanical_efficiency
    inertia = (c.rotor_inertia_kg_m2+c.reel_inertia_kg_m2)*accel/r
    torque = load + inertia + c.extra_shaft_friction_Nm
    return dict(tension_N=tension, cable_speed_m_s=speed, cable_acceleration_magnitude_m_s2=accel,
                cable_per_revolution_m=2*math.pi*r, rpm=omega*60/(2*math.pi),
                angular_accel_magnitude_rad_s2=accel/r, load_torque_Nm=load,
                inertia_torque_Nm=inertia, total_torque_bound_Nm=torque,
                sizing_torque_Nm=torque*c.torque_margin_factor,
                cable_power_W=tension*speed, shaft_power_bound_W=torque*omega,
                electrical_power_estimate_W=torque*omega/c.motor_driver_efficiency)


def draw(c, fig):
    import numpy as np
    result = calculate(c)
    ds = np.linspace(c.sweep_min_diameter_mm, c.sweep_max_diameter_mm, int(c.sweep_points))
    rows = [calculate(c, d) for d in ds]
    fig.clear()
    axes = fig.subplots(2, 1)
    axes[0].plot(ds, [r['rpm'] for r in rows], color='#2563eb')
    axes[0].set(ylabel='Required RPM', title='Reel diameter tradeoff — fixed inertia')
    for key, label in [('load_torque_Nm', 'Cable load'), ('inertia_torque_Nm', 'Rotating inertia'),
                       ('total_torque_bound_Nm', 'Total bound'), ('sizing_torque_Nm', 'With sizing margin')]:
        axes[1].plot(ds, [r[key] for r in rows], label=label)
    if c.motor_curve_csv.strip():
        data = np.genfromtxt(c.motor_curve_csv, delimiter=',', names=True)
        if data.dtype.names != ('rpm', 'torque_Nm'):
            raise ValueError('Curve CSV header must be rpm,torque_Nm')
        data = np.atleast_1d(data)
        x, y = data['rpm'], data['torque_Nm']
        if len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(x<0) or np.any(y<0):
            raise ValueError('Curve requires at least two finite nonnegative points.')
        order = np.argsort(x); x, y = x[order], y[order]
        if np.any(np.diff(x)<=0):
            raise ValueError('Curve RPM values must be unique.')
        available = np.interp([r['rpm'] for r in rows], x, y, left=np.nan, right=np.nan)
        axes[1].plot(ds, available, '--', label='Motor available torque (CSV)')
        result['available_motor_torque_Nm'] = float(np.interp(result['rpm'], x, y, left=np.nan, right=np.nan))
    axes[1].set(xlabel='Effective reel diameter (mm)', ylabel='Torque (N·m)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(c.reel_diameter_mm, color='gray', linestyle=':')
        ax.grid(alpha=.2)
    fig.tight_layout()
    return result


def gui():
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    root = tk.Tk(); root.title('Part C — editable motor and reel simulation'); root.geometry('1250x900')
    left = ttk.Frame(root, padding=12); left.pack(side='left', fill='y')
    right = ttk.Frame(root, padding=12); right.pack(side='right', fill='both', expand=True)
    ttk.Label(left, text='Edit assumptions, then Recalculate', font=('', 12, 'bold')).pack(anchor='w')
    ttk.Label(left, text='direct: enter cable loads | center: derive from Part B\nEfficiencies: fractions, e.g. 0.90 = 90%').pack(anchor='w', pady=8)
    holder = tk.Canvas(left, width=445, highlightthickness=0)
    scroll = ttk.Scrollbar(left, orient='vertical', command=holder.yview)
    scroll.pack(side='right', fill='y'); holder.pack(fill='both', expand=True)
    holder.configure(yscrollcommand=scroll.set)
    form = ttk.Frame(holder); holder.create_window((0,0), window=form, anchor='nw')
    form.bind('<Configure>', lambda e: holder.configure(scrollregion=holder.bbox('all')))
    inputs = {}
    for i, f in enumerate(fields(CONFIG)):
        ttk.Label(form, text=f.name.replace('_',' ')).grid(row=i, column=0, sticky='w', pady=4)
        var = tk.StringVar(value=str(getattr(CONFIG, f.name))); inputs[f.name]=var
        if f.name == 'mode':
            widget = ttk.Combobox(form, textvariable=var, values=('direct','center'), state='readonly', width=15)
        else:
            widget = ttk.Entry(form, textvariable=var, width=18)
        widget.grid(row=i, column=1, padx=8)
    fig = Figure(figsize=(7,5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=right); canvas.get_tk_widget().pack(fill='both', expand=True)
    output = tk.Text(right, height=12, font=('Courier',10)); output.pack(fill='x')
    def settings():
        values = {}
        for f in fields(CONFIG):
            raw = inputs[f.name].get()
            values[f.name] = type(getattr(CONFIG,f.name))(raw)
        c = Settings(**values); validate(c); return c
    def update():
        try:
            r = draw(settings(), fig); canvas.draw()
            output.delete('1.0',tk.END)
            output.insert(tk.END,'\n'.join(f'{k:43s} {v:.5g}' for k,v in r.items()))
            output.insert(tk.END,'\n\nAssumed loads; torque bound, not a verified motor requirement.\nElectrical estimate excludes standstill/idle losses.')
        except (ValueError, OSError, TypeError) as exc:
            messagebox.showerror('Check inputs', str(exc))
    def save():
        try:
            c=settings(); filename=filedialog.asksaveasfilename(defaultextension='.json')
            if filename: Path(filename).write_text(json.dumps(c.__dict__, indent=2))
        except (ValueError, OSError) as exc: messagebox.showerror('Save failed', str(exc))
    def load():
        filename=filedialog.askopenfilename(filetypes=[('Settings JSON','*.json')])
        if not filename: return
        try:
            c=Settings(**json.loads(Path(filename).read_text())); validate(c)
            for f in fields(c): inputs[f.name].set(str(getattr(c,f.name)))
            update()
        except (ValueError, TypeError, OSError) as exc: messagebox.showerror('Load failed',str(exc))
    buttons=ttk.Frame(right); buttons.pack(fill='x', pady=6)
    for title, command in [('Recalculate',update),('Save settings',save),('Load settings',load)]:
        ttk.Button(buttons,text=title,command=command).pack(side='left',padx=4)
    update(); root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Print results and save a plot without a window')
    parser.add_argument('--config', help='Load JSON settings (missing fields use defaults)')
    args = parser.parse_args()
    if args.config: CONFIG = Settings(**json.loads(Path(args.config).read_text()))
    if args.headless:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        figure = Figure(figsize=(8,7))
        print(json.dumps(draw(CONFIG, figure), indent=2))
        figure.savefig('part_c_results.png', dpi=160)
    else:
        gui()
