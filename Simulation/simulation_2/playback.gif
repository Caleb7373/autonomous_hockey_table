"""Run: python air_hockey_sim.py
Headless: python air_hockey_sim.py --headless --output results
Dependencies: numpy, matplotlib (Tkinter normally included with Windows Python).
See physics.py for the complete, commented equations and README.md for limits.
"""
import argparse
from dataclasses import fields, replace
from pathlib import Path
import json
import time
import threading
import queue
import numpy as np
from physics import (Settings,Motor,PRESETS,LABELS,COLORS,Model,default_motors,
                     simulate,summary,export_csv,save_config,load_config,validate)

EQUATIONS = '''MODEL / SIGN CONVENTIONS
All distances in metres, time in seconds, force in newtons.
Positive motor rotation winds the string IN. Forward is +y.
Four strings attach at an ideal nonrotating striker hub.
The puck is free: it has no strings attached.

1. GEOMETRY, EACH STRING i
  d_i = anchor_i - striker_position
  L_i = ||d_i||          e_i = d_i / L_i
  Ldot_i = -e_i dot striker_velocity
  l_i = l0_i - reel_radius_i * motor_angle_i
  extension_i = L_i - l_i
  extension_rate_i = Ldot_i + reel_radius_i * motor_speed_i

2. ACTUAL TENSION (NOT THE CONTROLLER TARGET)
  If extension <= 0: T = 0 (slack).
  Otherwise: T = max(0, k*extension + c*extension_rate).
  k: effective string stiffness [N/m]. c: damping [N s/m].
  Strings pull; they cannot push. Peak values depend on k and c.

3. STRIKER / PUCK MOTION
  m_s*a_s = sum(T_i*e_i) + drag - contact_force + wall_force
  m_p*a_p = drag + contact_force + wall_force
  drag(v) = -F_drag * v / sqrt(v dot v + smoothing_speed^2)
  This smoothed drag law approximates sliding resistance near zero.

4. CONTACT WITH THE PUCK
  overlap = R_s + R_p - distance_between_centres
  normal = (puck_position-striker_position)/distance
  closing_speed = (striker_velocity-puck_velocity) dot normal
  F_n = max(0, k_contact*overlap + c_contact*closing_speed)
  F_n = 0 when overlap <= 0. Forces are equal and opposite.
  No fixed restitution coefficient: rebound comes from stiffness/damping.
  Wall contacts use the same spring/damper idea. Goals are openings.

5. MOTOR / REEL DYNAMICS (DIRECT DRIVE)
  J = rotor_inertia + reel_inertia
  J*omega_dot = applied_torque - T*r - b*omega - loss_torque
  loss_torque = T*r*(1/eta-1)*tanh(omega/omega_smoothing)
  This is a passive friction approximation for mechanical efficiency.
  At winding speed, it approaches load torque T*r/eta.
  Motor torque approaches its limited command over response_time.
  Applied torque is also clipped to the instantaneous speed envelope.
  There is no artificial clamp on actual RPM: a cable can backdrive a reel.

6. CONTROLLER (ASSUMED, IDEAL FEEDBACK)
  a_cmd = a_ref + Kp*(p_ref-p) + Kv*(v_ref-v)
  a_cmd magnitude is limited; F_des = m_s*a_cmd - drag.
  Allocate T_des to minimize:
    ||sum(T_des_i*e_i)-F_des||^2 + lambda*||T_des-T_min||^2
    subject to T_min <= T_des_i <= T_max.
  This is iterative projected gradient; force residual is recorded.
  omega_ff = (e dot v_ref)/r
  alpha_ff = [e dot a_ref - (v_ref^2-(e dot v_ref)^2)/L]/r
  tau_cmd = T_des*r + J*alpha_ff + Kt*r*(T_des-T_actual)
            + Kspeed*(omega_ff-omega) + friction_compensation
  The motor may be too weak to achieve that command. T_actual can go
  below T_min or above T_max: those are controller targets, not clamps.

7. EXAMPLE MOTOR ENVELOPES (NOT VERIFIED PRODUCT DATA)
  n = absolute shaft RPM; n0 = characteristic_speed_rpm
  Stepper: tau_limit = peak_torque / (1 + (n/n0)^2)
  DC:      tau_limit = peak_torque * max(0, 1-n/n0)
  Servo:   tau_limit = peak_torque * min(1, n0/max(n,tiny))
  Custom:  linear interpolation of CSV columns rpm,torque_Nm.
  All envelopes return zero above maximum_speed_rpm.
  Custom curves return zero outside their measured RPM range and are
  also capped by peak_torque. They must start at 0 RPM.
  Same symmetric envelope is assumed for winding and braking.
  Continuous torque is a reporting threshold, not a thermal prediction.

8. TIME / LIMITATIONS
  RK4 integration recomputes all physical forces at each substage.
  Controller commands are held between control updates.
  Tension CSV contains EVERY integration sample, default 0.1 ms.
  Playback/plots show fewer samples; they do not change the physics.
  Constant reel radius; no line layering, spin, tipping, or finite
  pulley geometry. No electrical current, regeneration, heating,
  encoder noise, real sensor delay, or stepper missed-step model.
  Stiffness, damping, gains and preset motors are assumptions.
  Results are exploratory predictions, not validated hardware loads.
'''


class View:
    """Same drawing code powers the GUI and the headless plot for visual QA."""
    def __init__(self,figure,d):
        from matplotlib.patches import Circle,Rectangle
        self.fig=figure;self.d=d;self.s=s=d['settings'];figure.clear()
        gs=figure.add_gridspec(2,2,width_ratios=[1,1.38],hspace=.40,wspace=.34,
                              left=.06,right=.98,top=.91,bottom=.17)
        self.table=figure.add_subplot(gs[:,0]);self.tension=figure.add_subplot(gs[0,1]);self.speed=figure.add_subplot(gs[1,1])
        ax=self.table;ax.set_aspect('equal');ax.set_facecolor('#f3f7fb')
        ax.set_xlim(-.12,s.table_width_m+.12);ax.set_ylim(-.09,s.table_length_m+.12)
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Top view · opponent at top',fontsize=11)
        ax.add_patch(Rectangle((0,0),s.table_width_m,s.table_length_m,facecolor='#f8fbff',edgecolor='#65768b',lw=2))
        ax.axhline(s.robot_zone_length_m,color='#b8c6d6',ls=':',lw=1)
        for y in (0,s.table_length_m):
            ax.plot([s.table_width_m/2-s.goal_width_m/2,s.table_width_m/2+s.goal_width_m/2],[y,y],lw=6,color='#26a69a')
        a=d['anchors'];self.cables=[];self.spokes=[];self.labels=[]
        for i in range(4):
            ax.add_patch(Circle(a[i],.020,facecolor=COLORS[i],edgecolor='white',lw=1,zorder=6))
            self.spokes.append(ax.plot([],[],color='white',lw=1.4,zorder=7)[0])
            self.cables.append(ax.plot([],[],color=COLORS[i],lw=1.7,zorder=2)[0])
            dx=-.027 if i%2==0 else .027
            self.labels.append(ax.text(a[i,0]+dx,a[i,1],f'M{i+1}',ha='right' if i%2==0 else 'left',va='center',fontsize=8,color=COLORS[i]))
        self.path=ax.plot(d['reference'][:,0],d['reference'][:,1],color='#a5b5c6',ls='--',lw=1)[0]
        self.striker=Circle(d['state'][0,:2],s.striker_radius_m,facecolor='#253c5b',edgecolor='white',lw=2,zorder=4);ax.add_patch(self.striker)
        self.hub=Circle(d['state'][0,:2],s.striker_radius_m*.18,color='white',zorder=5);ax.add_patch(self.hub)
        self.puck=Circle(d['state'][0,4:6],s.puck_radius_m,facecolor='#ef5464',edgecolor='white',lw=1.5,zorder=4);ax.add_patch(self.puck)
        self.refdot=ax.plot([],[],'+',color='#708399',markersize=10)[0]
        self.trail=ax.plot([],[],color='#ef5464',alpha=.25,lw=1)[0]
        self.info=figure.text(.04,.96,'',fontsize=12,fontweight='bold',color='#253c5b')
        self.caption=figure.text(.04,.03,'Strings attach to striker hub. Presets and physical parameters are illustrative.',fontsize=8,color='#66758b')
        self.badges=ax.text(.5,-.055,'',transform=ax.transAxes,ha='center',va='top',fontsize=8)
        t=d['t'];stride=max(1,len(t)//5000)
        for i in range(4):self.tension.plot(t[::stride],d['tension'][::stride,i],color=COLORS[i],lw=1,
                                          ls='-' if i%2==0 else '--',label=f'M{i+1}')
        self.tension.set(xlabel='Time (s)',ylabel='Actual tension (N)',title='All four strings · full run')
        self.tension.legend(loc='upper right',fontsize=8,ncol=4);self.tension.grid(alpha=.15)
        self.speed.plot(t[::stride],np.linalg.norm(d['state'][::stride,2:4],axis=1),color='#253c5b',label='Striker')
        self.speed.plot(t[::stride],np.linalg.norm(d['state'][::stride,6:8],axis=1),color='#ef5464',label='Puck')
        self.speed.set(xlabel='Time (s)',ylabel='Speed (m/s)',title='Motion and impact')
        self.speed.legend(fontsize=8,loc='upper right');self.speed.grid(alpha=.15)
        for ax in [self.tension,self.speed]:
            ax.set_xlim(0,t[-1]);ax.tick_params(labelsize=8)
            for hit in d['hit_times']:ax.axvline(hit,ls=':',color='#ef5464',lw=.9,alpha=.7)
        self.cursors=[ax.axvline(0,color='#111827',lw=1,alpha=.6) for ax in [self.tension,self.speed]]
        self.dots=[self.tension.plot([],[],'o',color=COLORS[i],ms=4)[0] for i in range(4)]
        self.update(0)

    def update(self,index):
        d=self.d;s=self.s;i=int(np.clip(index,0,len(d['t'])-1));y=d['state'][i];t=d['t'][i]
        self.striker.center=y[:2];self.hub.center=y[:2];self.puck.center=y[4:6]
        self.refdot.set_data([d['reference'][i,0]],[d['reference'][i,1]])
        lo=max(0,i-int(.2/s.integration_step_s));step=max(1,(i-lo)//100)
        self.trail.set_data(d['state'][lo:i+1:step,4],d['state'][lo:i+1:step,5])
        values=[]
        for j in range(4):
            a=d['anchors'][j];p=y[:2];T=d['tension'][i,j]
            self.cables[j].set_data([a[0],p[0]],[a[1],p[1]])
            self.cables[j].set_linestyle('--' if T<=1e-6 else '-')
            self.cables[j].set_alpha(.35 if T<=1e-6 else .9)
            self.cables[j].set_linewidth(1+min(T/12,2))
            angle=y[8+j];end=a+.017*np.array([np.cos(angle),np.sin(angle)])
            self.spokes[j].set_data([a[0],end[0]],[a[1],end[1]])
            self.dots[j].set_data([t],[T]);values.append(f'M{j+1}: {T:5.2f} N')
        self.badges.set_text('   '.join(values[:2])+'\n'+'   '.join(values[2:]))
        for line in self.cursors:line.set_xdata([t,t])
        event='GOAL' if d['scored'][i] else ('CONTACT' if d['contact_force'][i]>0 else 'RUN')
        self.info.set_text(f'{t:0.3f} s   |   {event}   |   puck {np.linalg.norm(y[6:8]):.2f} m/s')


def numeric_breakdown(d,i,motor=0):
    s=d['settings'];m=d['motors'][motor];y=d['state'][i];r=m.reel_diameter_mm/2000
    L=float(np.linalg.norm(d['anchors'][motor]-y[:2]));e=(d['anchors'][motor]-y[:2])/L
    l=d['l0'][motor]-r*y[8+motor];x=L-l;xdot=float(-e@y[2:4]+r*y[12+motor])
    limit=d['torque_limit'][i,motor];T=d['tension'][i,motor];J=m.rotor_inertia_kg_m2+m.reel_inertia_kg_m2
    loss=T*r*(1/m.mechanical_efficiency-1)*np.tanh(y[12+motor]/s.loss_smoothing_rad_s)
    alpha=(d['torque'][i,motor]-T*r-m.viscous_friction_Nms*y[12+motor]-loss)/J
    return (f'TIME {d["t"][i]:.4f} s  |  M{motor+1}: {LABELS[motor]}\n'
            f'{m.name} ({m.model}); r={r:.4f} m\n'
            f'L={L:.6f} m   unstretched l={l:.6f} m\n'
            f'Extension=L-l={x:.6f} m ({x*1000:.3f} mm)\n'
            f'Extension rate={xdot:.5f} m/s\n'
            f'T=max(0, {s.cable_stiffness_N_m:g}*{x:.6f} + {s.cable_damping_Ns_m:g}*{xdot:.5f})\n'
            f'  = {T:.4f} N'+(' [slack: extension <= 0]' if x<=0 else '')+'\n'
            f'Tension target={d["target_tension"][i,motor]:.4f} N\n'
            f'RPM={y[12+motor]*60/(2*np.pi):.2f} (positive=winding in)\n'
            f'Torque requested={d["torque_command"][i,motor]:.4f} N m\n'
            f'Torque available=+/-{limit:.4f}; applied={d["torque"][i,motor]:.4f} N m\n'
            f'alpha=(applied - T*r - viscous - efficiency_loss)/J\n'
            f'     = ({d["torque"][i,motor]:.4f} - {T*r:.4f} - {m.viscous_friction_Nms*y[12+motor]:.4f} - {loss:.4f})/{J:.6g}\n'
            f'     = {alpha:.2f} rad/s^2\n'
            f'Contact force={d["contact_force"][i]:.3f} N\n'
            f'Controller force residual={d["allocation_residual"][i]:.4f} N')


def headless(s,motors,outdir):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    d=simulate(s,motors);export_csv(d,out/'telemetry.csv');save_config(out/'settings.json',s,motors)
    report=summary(d);(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig=Figure(figsize=(11,7),dpi=140);v=View(fig,d)
    index=int(np.argmax(d['contact_force'])) if d['contact_force'].max()>0 else len(d['t'])//2
    v.update(index);fig.savefig(out/'simulation.png',dpi=150)
    print(json.dumps(report,indent=2));print('Saved outputs to',out.resolve());return d


def gui(initial_settings,initial_motors):
    import tkinter as tk
    from tkinter import ttk,messagebox,filedialog
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    root=tk.Tk();root.title('Air Hockey · motors, strings & impact');root.geometry('1430x960');root.minsize(1120,780)
    ttk.Style().configure('TButton',padding=5)
    outer=ttk.Panedwindow(root,orient='horizontal');outer.pack(fill='both',expand=True)
    left=ttk.Frame(outer,padding=10);right=ttk.Frame(outer,padding=8);outer.add(left,weight=0);outer.add(right,weight=1)
    ttk.Label(left,text='Air hockey simulation',font=('',16,'bold')).pack(anchor='w')
    ttk.Label(left,text='Edit inputs, then Simulate. All presets are examples.\nStrings connect motors to the striker; puck is free.',wraplength=380).pack(anchor='w',pady=(4,8))
    settings_book=ttk.Notebook(left);settings_book.pack(fill='both',expand=True)
    variables={};motor_vars=[];busy=False;data=None;view=None;playing=False;current=0;play_started=0.;play_from=0.;scrub_internal=False
    events=queue.Queue();status=tk.StringVar(value='Ready');progress=tk.DoubleVar(value=0.)
    def scrollable(parent):
        page=ttk.Frame(parent);can=tk.Canvas(page,width=390,highlightthickness=0)
        bar=ttk.Scrollbar(page,orient='vertical',command=can.yview);bar.pack(side='right',fill='y');can.pack(side='left',fill='both',expand=True)
        can.configure(yscrollcommand=bar.set);body=ttk.Frame(can);wid=can.create_window((0,0),window=body,anchor='nw')
        body.bind('<Configure>',lambda e:can.configure(scrollregion=can.bbox('all')))
        can.bind('<Configure>',lambda e:can.itemconfigure(wid,width=e.width))
        can.bind('<Enter>',lambda e:can.bind_all('<MouseWheel>',lambda ev:can.yview_scroll(-int(ev.delta/120),'units')))
        can.bind('<Leave>',lambda e:can.unbind_all('<MouseWheel>'))
        return page,body
    def add_field(body,row,f,value,dest):
        name=f.name.replace('_',' ');ttk.Label(body,text=name,wraplength=248).grid(row=row,column=0,sticky='w',pady=4,padx=(2,6))
        var=tk.StringVar(value=str(value));dest[f.name]=var
        if f.name=='model':w=ttk.Combobox(body,textvariable=var,values=('stepper','dc','servo','custom'),state='readonly',width=12)
        else:w=ttk.Entry(body,textvariable=var,width=14)
        w.grid(row=row,column=1,sticky='e',pady=4);body.columnconfigure(0,weight=1)
    # Group by explicit boundary names so future fields don't silently vanish.
    all_fields=list(fields(initial_settings));boundaries=[0, next(i for i,f in enumerate(all_fields) if f.name=='cable_stiffness_N_m'),
        next(i for i,f in enumerate(all_fields) if f.name=='position_gain_s2'),next(i for i,f in enumerate(all_fields) if f.name=='duration_s'),len(all_fields)]
    for idx,title in enumerate(['Motion','Strings','Control','Time']):
        page,body=scrollable(settings_book);settings_book.add(page,text=title)
        for row,f in enumerate(all_fields[boundaries[idx]:boundaries[idx+1]]):add_field(body,row,f,getattr(initial_settings,f.name),variables)
    motorpage=ttk.Frame(settings_book);settings_book.add(motorpage,text='Motors')
    motorsbook=ttk.Notebook(motorpage);motorsbook.pack(fill='both',expand=True)
    for j,m in enumerate(initial_motors):
        page,body=scrollable(motorsbook);motorsbook.add(page,text=f'M{j+1}')
        dest={};motor_vars.append(dest)
        ttk.Label(body,text=LABELS[j],font=('',11,'bold')).grid(row=0,column=0,columnspan=2,sticky='w',pady=5)
        preset=tk.StringVar(value='Choose example preset')
        combo=ttk.Combobox(body,textvariable=preset,values=list(PRESETS),state='readonly',width=29);combo.grid(row=1,column=0,columnspan=2,sticky='ew',pady=5)
        def setpreset(ev,which=j,selection=preset):
            for f in fields(Motor):motor_vars[which][f.name].set(str(getattr(PRESETS[selection.get()],f.name)))
        combo.bind('<<ComboboxSelected>>',setpreset)
        for row,f in enumerate(fields(m),start=2):add_field(body,row,f,getattr(m,f.name),dest)
        def choose_curve(which=j):
            name=filedialog.askopenfilename(filetypes=[('Torque-speed CSV','*.csv')])
            if name:motor_vars[which]['curve_csv'].set(name);motor_vars[which]['model'].set('custom')
        ttk.Button(body,text='Load torque-speed CSV',command=choose_curve).grid(row=2+len(fields(m)),column=0,columnspan=2,pady=5)
    def copy_m1():
        for dest in motor_vars[1:]:
            for name,var in motor_vars[0].items():dest[name].set(var.get())
    ttk.Button(motorpage,text='Copy M1 settings to all four motors',command=copy_m1).pack(fill='x',pady=5)
    ttk.Label(motorpage,text='M1/M2 rear; M3/M4 front.\nTypes compare ideal feedback actuators, not step pulses.',wraplength=380).pack(pady=4)
    ttk.Label(left,textvariable=status,wraplength=390).pack(fill='x',pady=8)
    ttk.Progressbar(left,variable=progress,maximum=1).pack(fill='x')
    commands=ttk.Frame(left);commands.pack(fill='x',pady=8)
    mainbook=ttk.Notebook(right);mainbook.pack(fill='both',expand=True)
    animpage=ttk.Frame(mainbook);calcpage=ttk.Frame(mainbook);eqpage=ttk.Frame(mainbook);reportpage=ttk.Frame(mainbook)
    for p,title in [(animpage,'Animation & tension'),(calcpage,'Live calculation'),(eqpage,'Equations'),(reportpage,'Run summary')]:mainbook.add(p,text=title)
    fig=Figure(figsize=(9,6),dpi=100);canvas=FigureCanvasTkAgg(fig,master=animpage);canvas.get_tk_widget().pack(fill='both',expand=True)
    eqtext=tk.Text(eqpage,wrap='word',font=('Consolas',11),padx=14,pady=12);eqscroll=ttk.Scrollbar(eqpage,command=eqtext.yview);eqscroll.pack(side='right',fill='y');eqtext.pack(fill='both',expand=True);eqtext.configure(yscrollcommand=eqscroll.set);eqtext.insert('1.0',EQUATIONS);eqtext.configure(state='disabled')
    inspect=tk.IntVar(value=0);calcbuttons=ttk.Frame(calcpage);calcbuttons.pack(fill='x',padx=12,pady=8)
    calctext=tk.Text(calcpage,font=('Consolas',11),wrap='word',padx=14,pady=12);calctext.pack(fill='both',expand=True)
    reporttext=tk.Text(reportpage,font=('Consolas',10),wrap='word',padx=14,pady=12);reporttext.pack(fill='both',expand=True)
    def display(index):
        nonlocal current,scrub_internal
        if data is None:return
        current=int(np.clip(index,0,len(data['t'])-1));view.update(current);canvas.draw_idle()
        calctext.delete('1.0',tk.END);calctext.insert('1.0',numeric_breakdown(data,current,inspect.get()))
        scrub_internal=True;scrub.set(data['t'][current]);scrub_internal=False
        stamp.set(f'{data["t"][current]:.4f} / {data["t"][-1]:.3f} s')
    for j in range(4):ttk.Radiobutton(calcbuttons,text=f'M{j+1} {LABELS[j]}',variable=inspect,value=j,command=lambda:display(current)).pack(side='left',padx=6)
    playback=ttk.Frame(right);playback.pack(fill='x',pady=(6,0));stamp=tk.StringVar(value='No run yet')
    speed=tk.StringVar(value='0.5')
    def toggle():
        nonlocal playing,play_started,play_from
        if data is None:return
        if playing:playing=False;playbtn.configure(text='Play');return
        if current>=len(data['t'])-1:display(0)
        playing=True;play_started=time.perf_counter();play_from=data['t'][current];playbtn.configure(text='Pause')
    def rewind():
        nonlocal playing
        playing=False;playbtn.configure(text='Play');display(0)
    playbtn=ttk.Button(playback,text='Play',command=toggle);playbtn.pack(side='left')
    ttk.Button(playback,text='Rewind',command=rewind).pack(side='left',padx=4)
    ttk.Label(playback,text='Playback speed').pack(side='left',padx=(12,4))
    rate=ttk.Combobox(playback,textvariable=speed,values=['0.1','0.25','0.5','1','2'],state='readonly',width=5);rate.pack(side='left')
    def rate_changed(ev):
        nonlocal play_started,play_from
        if data is not None:play_started=time.perf_counter();play_from=data['t'][current]
    rate.bind('<<ComboboxSelected>>',rate_changed)
    ttk.Label(playback,textvariable=stamp).pack(side='right')
    def scrubbed(value):
        nonlocal playing
        if scrub_internal or data is None:return
        playing=False;playbtn.configure(text='Play');display(np.searchsorted(data['t'],float(value)))
    scrub=ttk.Scale(right,from_=0,to=initial_settings.duration_s,command=scrubbed);scrub.pack(fill='x',pady=6)
    def read_inputs():
        vals={f.name:type(getattr(initial_settings,f.name))(variables[f.name].get()) for f in fields(initial_settings)}
        s=Settings(**vals);ms=[]
        for dest in motor_vars:
            ms.append(Motor(**{f.name:type(getattr(Motor(),f.name))(dest[f.name].get()) for f in fields(Motor)}))
        validate(s,ms);return s,ms
    def run():
        nonlocal busy,playing
        if busy:return
        try:s,ms=read_inputs();Model(s,ms)
        except (ValueError,OSError,TypeError) as e:messagebox.showerror('Check inputs',str(e));return
        busy=True;playing=False;playbtn.configure(text='Play');runbtn.configure(state='disabled');progress.set(0);status.set('Calculating motion and tension…')
        def worker():
            try:events.put(('done',simulate(s,ms,lambda p:events.put(('progress',p)))))
            except Exception as e:events.put(('error',str(e)))
        threading.Thread(target=worker,daemon=True).start()
    def save_settings():
        try:
            s,ms=read_inputs();name=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('Settings','*.json')])
            if name:save_config(name,s,ms)
        except (ValueError,OSError,TypeError) as e:messagebox.showerror('Settings',str(e))
    def load_settings():
        name=filedialog.askopenfilename(filetypes=[('Settings','*.json')])
        if not name:return
        try:
            s,ms=load_config(name)
            for f in fields(s):variables[f.name].set(str(getattr(s,f.name)))
            for j,m in enumerate(ms):
                for f in fields(m):motor_vars[j][f.name].set(str(getattr(m,f.name)))
            status.set('Settings loaded. Click Simulate to apply.')
        except (ValueError,OSError,TypeError,KeyError) as e:messagebox.showerror('Settings',str(e))
    def csv():
        if data is None:return
        name=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='tensions_and_motion.csv',filetypes=[('CSV','*.csv')])
        if name:
            try:export_csv(data,name);status.set(f'Saved {len(data["t"]):,} time samples.')
            except OSError as e:messagebox.showerror('Export',str(e))
    runbtn=ttk.Button(commands,text='Simulate',command=run);runbtn.pack(side='left',fill='x',expand=True)
    ttk.Button(commands,text='Export CSV',command=csv).pack(side='left',padx=4)
    files=ttk.Frame(left);files.pack(fill='x');ttk.Button(files,text='Save settings',command=save_settings).pack(side='left',expand=True,fill='x');ttk.Button(files,text='Load settings',command=load_settings).pack(side='left',expand=True,fill='x',padx=4)
    def loop():
        nonlocal busy,data,view,playing
        try:
            while True:
                kind,payload=events.get_nowait()
                if kind=='progress':progress.set(payload)
                elif kind=='error':busy=False;runbtn.configure(state='normal');status.set('Run failed. Check inputs.');messagebox.showerror('Simulation',payload)
                else:
                    data=payload;view=View(fig,data);busy=False;runbtn.configure(state='normal');progress.set(1)
                    scrub.configure(to=data['t'][-1]);report=summary(data)
                    reporttext.delete('1.0',tk.END);reporttext.insert('1.0',json.dumps(report,indent=2)+'\n\nAll loads are model predictions. Continuous-torque time is a diagnostic, not a thermal safety result.\n\nMotor profiles:\n'+json.dumps([m.__dict__ for m in data['motors']],indent=2))
                    status.set(f'Run complete: {len(data["t"]):,} samples. '+('Puck struck.' if data['hit_times'] else 'No puck contact.'))
                    display(0);toggle()
        except queue.Empty:pass
        if playing and data is not None:
            t=play_from+(time.perf_counter()-play_started)*float(speed.get());display(np.searchsorted(data['t'],t))
            if t>=data['t'][-1]:playing=False;playbtn.configure(text='Play')
        root.after(33,loop)
    root.after(100,run);root.after(33,loop);root.mainloop()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',help='Load saved JSON configuration')
    parser.add_argument('--headless',action='store_true',help='Calculate and export without opening a window')
    parser.add_argument('--output',default='results',help='Headless output directory')
    args=parser.parse_args();s,ms=load_config(args.config) if args.config else (Settings(),default_motors())
    if args.headless:headless(s,ms,args.output)
    else:gui(s,ms)
