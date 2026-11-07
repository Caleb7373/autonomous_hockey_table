"""Planar four-cable robot, motor reels, compliant puck contact.

Units: SI throughout, except explicit *_mm and *_rpm fields.
No preset represents a measured commercial motor. See README.md for assumptions.

Sign convention: theta/omega positive WINDING IN. Each cable attaches to the
striker CENTER via an ideal nonrotating hub (torque/tipping/spin are excluded).
Cable anchors are fixed exit points colocated with motor icons.

MODEL EQUATIONS (also displayed in the application):
  d_i = anchor_i - p; L_i = norm(d_i); e_i = d_i/L_i
  Ldot_i = -e_i dot v
  l_i = l0_i - r_i*theta_i             [unstretched paid-out length]
  x_i = L_i-l_i; xdot_i=Ldot_i+r_i*omega_i
  T_i = max(0, k*x_i+c*xdot_i) if x_i>0 else 0
  m_s*a_s = sum(T_i*e_i) + F_drag - F_contact + F_walls
  J_i*omega_dot_i = tau_i - T_i*r_i - b_i*omega_i - tau_loss_i
  tau_loss_i=T_i*r_i*(1/eta_i-1)*tanh(omega_i/omega_smooth)
  tau_dot_i=(clip(tau_command_i, +/- torque_limit_i(rpm))-tau_i)/response_s
  puck force = +F_contact + F_drag + F_walls

Cable dashpots act only while stretched; cables never push. Efficiency is a
LUMPED DISSIPATIVE FRICTION APPROXIMATION, not an electrical motor model. At
appreciable winding speed it gives steady tau = Tr/eta. During payout it remains
dissipative; it is not a claim about actual regenerative drive efficiency.

Contact: overlap=R_s+R_p-distance; overlap_dot=(v_s-v_p).normal.
  F_n=max(0,k_contact*overlap+c_contact*overlap_dot) if overlap>0 else 0
Equal and opposite forces act throughout finite-duration contact. Damping is
an input, NOT an exact restitution coefficient. No collision impulse is imposed.

Trajectory: straight trapezoidal/triangular speed profile from start to target.
Controller: Cartesian PD + acceleration feedforward -> bounded tension targets
-> reel torque commands with tension and speed feedback -> torque/speed limit.
All motor families use an ideal feedback drive with finite torque response.
Stepper microsteps, detent torque, missed steps, and electrical phase dynamics
are NOT modeled. Curves compare actuator envelopes, not complete drive systems.
"""
from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path
import json
import math
import numpy as np

LABELS = ['Rear left', 'Rear right', 'Front left', 'Front right']
COLORS = ['#3478f6', '#8b5cf6', '#ea8b24', '#15a28b']


@dataclass
class Settings:
    # TABLE AND MOTION: forward is +y, right is +x; metres.
    table_width_m: float = .720
    table_length_m: float = 1.440
    robot_zone_length_m: float = .720
    anchor_inset_m: float = .025
    goal_width_m: float = .240
    striker_mass_kg: float = .200
    striker_radius_m: float = .045
    puck_mass_kg: float = .040
    puck_radius_m: float = .032
    striker_start_x_m: float = .360
    striker_start_y_m: float = .200
    target_x_m: float = .360
    target_y_m: float = .640
    puck_start_x_m: float = .360
    puck_start_y_m: float = .510
    puck_initial_vx_m_s: float = 0.
    puck_initial_vy_m_s: float = 0.
    target_speed_m_s: float = 2.
    target_acceleration_m_s2: float = 10.
    start_delay_s: float = .150
    striker_drag_N: float = 1.
    puck_drag_N: float = .020  # .040 kg * .50 m/s^2, previous illustrative loss
    drag_smoothing_m_s: float = .020
    # STRING AND CONTACT: constant stiffness is an effective measured parameter.
    cable_stiffness_N_m: float = 2500.
    cable_damping_Ns_m: float = 1.5
    minimum_target_tension_N: float = 4.
    maximum_target_tension_N: float = 35.
    contact_stiffness_N_m: float = 20000.
    contact_damping_Ns_m: float = 6.
    wall_stiffness_N_m: float = 20000.
    wall_damping_Ns_m: float = 12.
    # CONTROL: every setting below is an assumed gain, not a measured tuning.
    position_gain_s2: float = 180.
    velocity_gain_s: float = 24.
    acceleration_command_limit_m_s2: float = 40.
    tension_feedback_gain: float = .75
    reel_speed_gain_Nms: float = .008
    allocation_regularization: float = .0001
    allocation_iterations: int = 50
    # TIME: tension is recorded EVERY integration step, not only animation frames.
    duration_s: float = 1.4
    integration_step_s: float = .0001
    control_period_s: float = .001
    loss_smoothing_rad_s: float = 1.


@dataclass
class Motor:
    name: str = 'Illustrative medium stepper'
    model: str = 'stepper'  # stepper, dc, servo, custom
    peak_torque_Nm: float = .9
    continuous_torque_Nm: float = .45  # reporting threshold only, no thermal model
    characteristic_speed_rpm: float = 550.  # stepper knee/DC no-load/servo base
    maximum_speed_rpm: float = 2500.  # torque envelope ends here, not a speed clamp
    rotor_inertia_kg_m2: float = .00010
    reel_inertia_kg_m2: float = .00003
    reel_diameter_mm: float = 90.
    mechanical_efficiency: float = .90
    viscous_friction_Nms: float = .00008
    response_time_s: float = .003
    curve_csv: str = ''  # only used for custom; columns rpm,torque_Nm, starting at 0


PRESETS = {
    'Medium stepper (example)': Motor(),
    'Small stepper (example)': Motor('Illustrative small stepper','stepper', .35,.18,400,2200,.000025,.00003,90,.90,.00005,.003),
    'DC motor (example)': Motor('Illustrative feedback DC motor','dc', .40,.12,3000,3000,.000012,.00003,45,.90,.00004,.002),
    'Servo (example)': Motor('Illustrative servo drive','servo', .8,.35,1200,4000,.00004,.00003,60,.92,.00004,.0015),
}


def default_motors():
    return [replace(PRESETS['Medium stepper (example)']) for _ in range(4)]


def anchors(s):
    b=s.anchor_inset_m
    return np.array([[b,b],[s.table_width_m-b,b],
                     [b,s.robot_zone_length_m-b],
                     [s.table_width_m-b,s.robot_zone_length_m-b]])


def validate(s, motors):
    if len(motors)!=4: raise ValueError('Exactly four motors are required.')
    for obj in [s,*motors]:
        for f in fields(obj):
            v=getattr(obj,f.name)
            if isinstance(v,(int,float)) and not np.isfinite(v):
                raise ValueError(f'{f.name} must be finite.')
    nonnegative=['start_delay_s','striker_drag_N','puck_drag_N','cable_damping_Ns_m',
                 'minimum_target_tension_N','contact_damping_Ns_m','wall_damping_Ns_m',
                 'position_gain_s2','velocity_gain_s','tension_feedback_gain',
                 'reel_speed_gain_Nms','allocation_regularization']
    signed=['puck_initial_vx_m_s','puck_initial_vy_m_s']
    for f in fields(s):
        v=getattr(s,f.name)
        if f.name in signed: continue
        if v<0 or (f.name not in nonnegative and v==0):
            raise ValueError(f'{f.name} must be '+('nonnegative.' if f.name in nonnegative else 'positive.'))
    if s.maximum_target_tension_N<s.minimum_target_tension_N:
        raise ValueError('Maximum target tension must be at least the minimum.')
    if not (0<s.robot_zone_length_m<=s.table_length_m): raise ValueError('Robot zone must fit on table.')
    if not 0<s.anchor_inset_m<min(s.table_width_m,s.robot_zone_length_m)/2:
        raise ValueError('Anchor inset is too large.')
    if not 2*s.puck_radius_m<s.goal_width_m<s.table_width_m:
        raise ValueError('Goal width must fit the puck and be smaller than table width.')
    for x,y,r,label in [(s.striker_start_x_m,s.striker_start_y_m,s.striker_radius_m,'Striker'),
                         (s.target_x_m,s.target_y_m,s.striker_radius_m,'Target'),
                         (s.puck_start_x_m,s.puck_start_y_m,s.puck_radius_m,'Puck')]:
        if not (r<x<s.table_width_m-r and r<y<s.table_length_m-r):
            raise ValueError(label+' must be fully inside the table.')
    if not (s.anchor_inset_m<s.target_x_m<s.table_width_m-s.anchor_inset_m and
            s.anchor_inset_m<s.target_y_m<s.robot_zone_length_m-s.anchor_inset_m):
        raise ValueError('Target must be inside the anchor rectangle.')
    if np.hypot(s.striker_start_x_m-s.puck_start_x_m,s.striker_start_y_m-s.puck_start_y_m)<=s.striker_radius_m+s.puck_radius_m:
        raise ValueError('Striker and puck cannot initially overlap.')
    if s.control_period_s<s.integration_step_s: raise ValueError('Control period cannot be shorter than integration step.')
    if s.duration_s/s.integration_step_s>500000: raise ValueError('Limit a run to 500,000 integration steps.')
    if int(s.allocation_iterations)!=s.allocation_iterations or not 1<=s.allocation_iterations<=1000:
        raise ValueError('Allocation iterations must be an integer from 1 to 1000.')
    ratio=s.control_period_s/s.integration_step_s
    if abs(ratio-round(ratio))>1e-6: raise ValueError('Control period must be an integer multiple of the integration step.')
    for m in motors:
        if m.model not in ('stepper','dc','servo','custom'): raise ValueError('Unknown motor model: '+m.model)
        if not 0<m.mechanical_efficiency<=1: raise ValueError('Efficiency must be in (0,1].')
        for key in ['peak_torque_Nm','continuous_torque_Nm','characteristic_speed_rpm','maximum_speed_rpm',
                    'rotor_inertia_kg_m2','reel_diameter_mm','response_time_s']:
            if getattr(m,key)<=0: raise ValueError(key+' must be positive.')
        if m.reel_inertia_kg_m2<0 or m.viscous_friction_Nms<0: raise ValueError('Inertia/friction cannot be negative.')
        if m.continuous_torque_Nm>m.peak_torque_Nm: raise ValueError('Continuous torque cannot exceed peak.')
    # Conservative explicit integration guard, not a proof of convergence.
    r=np.array([m.reel_diameter_mm/2000 for m in motors])
    j=np.array([m.rotor_inertia_kg_m2+m.reel_inertia_kg_m2 for m in motors])
    w2=4*s.cable_stiffness_N_m/s.striker_mass_kg + np.max(s.cable_stiffness_N_m*r*r/j)
    w2+=s.contact_stiffness_N_m*(1/s.striker_mass_kg+1/s.puck_mass_kg)+s.wall_stiffness_N_m/s.puck_mass_kg
    damp=4*s.cable_damping_Ns_m/s.striker_mass_kg+np.max(s.cable_damping_Ns_m*r*r/j)
    damp+=s.contact_damping_Ns_m*(1/s.striker_mass_kg+1/s.puck_mass_kg)+s.wall_damping_Ns_m/s.puck_mass_kg
    limit=min(.20/math.sqrt(w2),.20/max(damp,1), min(m.response_time_s for m in motors)/5)
    if s.integration_step_s>limit:
        raise ValueError(f'Integration step too large for these stiffness/inertia values. Use <= {limit:.2g} s and a compatible control period.')


def save_config(path,s,motors):
    Path(path).write_text(json.dumps({'settings':asdict(s),'motors':[asdict(m) for m in motors]},indent=2),encoding='utf-8')


def load_config(path):
    path=Path(path).resolve(); obj=json.loads(path.read_text(encoding='utf-8'))
    s=Settings(**obj['settings']); ms=[Motor(**m) for m in obj['motors']]
    for m in ms:
        if m.curve_csv and not Path(m.curve_csv).is_absolute(): m.curve_csv=str(path.parent/m.curve_csv)
    validate(s,ms); return s,ms


def drag(v,force,smoothing):
    return -force*v/math.sqrt(float(v@v)+smoothing*smoothing)


def contact(p,v,q,w,rs,rp,k,c):
    d=q-p; length=float(np.linalg.norm(d)); overlap=rs+rp-length
    if overlap<=0: return np.zeros(2),0.,0.
    n=d/max(length,1e-12) if length>1e-12 else np.array([0.,1.])
    closing=float((v-w)@n)
    fn=max(0.,k*overlap+c*closing)
    return fn*n,fn,overlap


def allocate(A,F,lo,hi,iterations,reg,initial=None):
    """Projected gradient: minimize .5||A T-F||²+.5*reg||T-lo||², lo<=T<=hi.
    Step 1/(||A||_F²+reg) is conservative. Residual is logged, never hidden.
    This returns a TARGET; actual tension still follows compliant dynamics.
    """
    T=np.full(4,lo) if initial is None else np.clip(initial,lo,hi).copy()
    step=1/(float(np.sum(A*A))+reg)
    for _ in range(int(iterations)):
        T=np.clip(T-step*(A.T@(A@T-F)+reg*(T-lo)),lo,hi)
    return T


class Model:
    def __init__(self,s,motors):
        validate(s,motors); self.s=s; self.motors=motors; self.anchors=anchors(s)
        self.r=np.array([m.reel_diameter_mm/2000 for m in motors])
        self.j=np.array([m.rotor_inertia_kg_m2+m.reel_inertia_kg_m2 for m in motors])
        self.eta=np.array([m.mechanical_efficiency for m in motors])
        self.b=np.array([m.viscous_friction_Nms for m in motors])
        self.response=np.array([m.response_time_s for m in motors])
        self.curves=[]
        for m in motors:
            curve=None
            if m.model=='custom':
                a=np.genfromtxt(m.curve_csv,delimiter=',',names=True)
                if a.dtype.names!=('rpm','torque_Nm'): raise ValueError('Curve header must be rpm,torque_Nm.')
                a=np.atleast_1d(a); x=a['rpm']; y=a['torque_Nm']; order=np.argsort(x);x=x[order];y=y[order]
                if len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or x[0]!=0 or np.any(np.diff(x)<=0) or np.any(y<0):
                    raise ValueError('Curve must have finite unique RPMs, start at 0 and contain nonnegative torque.')
                curve=(x,y)
            self.curves.append(curve)
        self.start=np.array([s.striker_start_x_m,s.striker_start_y_m]);self.target=np.array([s.target_x_m,s.target_y_m])
        delta=self.target-self.start;self.distance=float(np.linalg.norm(delta));self.direction=delta/max(self.distance,1e-12)
        self.vpeak=min(s.target_speed_m_s,math.sqrt(self.distance*s.target_acceleration_m_s2))
        self.ta=self.vpeak/s.target_acceleration_m_s2
        self.tc=max(0.,(self.distance-self.vpeak*self.ta)/max(self.vpeak,1e-12))
        L,e=self.geometry(self.start)
        self.Ttarget=allocate(e.T,np.zeros(2),s.minimum_target_tension_N,s.maximum_target_tension_N,2000,s.allocation_regularization)
        self.l0=L-self.Ttarget/s.cable_stiffness_N_m
        if np.any(self.l0<=0):raise ValueError('Initial pretension stretch exceeds cable length.')
        # State: p[0:2],v[2:4],q[4:6],w[6:8],theta[8:12],omega[12:16],tau[16:20]
        self.initial=np.zeros(20); self.initial[:2]=self.start
        self.initial[4:6]=[s.puck_start_x_m,s.puck_start_y_m]
        self.initial[6:8]=[s.puck_initial_vx_m_s,s.puck_initial_vy_m_s]
        self.initial[16:20]=np.minimum(self.Ttarget*self.r,self.limits(np.zeros(4)))
        self.command=self.initial[16:20].copy();self.force_residual=0.
        self.scored=False;self.goal=None

    def geometry(self,p):
        d=self.anchors-p;L=np.linalg.norm(d,axis=1)
        return L,d/np.maximum(L[:,None],1e-12)

    def reference(self,t):
        z=max(0.,t-self.s.start_delay_s); a=self.s.target_acceleration_m_s2
        if t<self.s.start_delay_s: d=v=acc=0.
        elif z<self.ta: d=.5*a*z*z;v=a*z;acc=a
        elif z<self.ta+self.tc: d=.5*a*self.ta**2+self.vpeak*(z-self.ta);v=self.vpeak;acc=0.
        elif z<2*self.ta+self.tc:
            u=z-self.ta-self.tc;d=.5*a*self.ta**2+self.vpeak*self.tc+self.vpeak*u-.5*a*u*u;v=self.vpeak-a*u;acc=-a
        else: d=self.distance;v=acc=0.
        return self.start+self.direction*d,self.direction*v,self.direction*acc

    def limits(self,omega):
        rpm=np.abs(omega)*60/(2*np.pi);out=np.empty(4)
        for i,m in enumerate(self.motors):
            n=rpm[i];n0=m.characteristic_speed_rpm
            if m.model=='stepper': value=m.peak_torque_Nm/(1+(n/n0)**2)
            elif m.model=='dc': value=m.peak_torque_Nm*max(0.,1-n/n0)
            elif m.model=='servo': value=m.peak_torque_Nm*min(1.,n0/max(n,1e-12))
            else: value=float(np.interp(n,*self.curves[i],left=0,right=0))
            out[i]=min(value,m.peak_torque_Nm) if n<m.maximum_speed_rpm else 0.
        return out

    def strings(self,y):
        L,e=self.geometry(y[:2]);l=self.l0-self.r*y[8:12]
        x=L-l;xdot=-e@y[2:4]+self.r*y[12:16]
        T=np.where(x>0,np.maximum(0,self.s.cable_stiffness_N_m*x+self.s.cable_damping_Ns_m*xdot),0.)
        return T,L,e,l,x,xdot

    def friction(self,T,omega):
        return T*self.r*(1/self.eta-1)*np.tanh(omega/self.s.loss_smoothing_rad_s)

    def control(self,t,y):
        s=self.s;p,v,a=self.reference(t);T,L,e,*_=self.strings(y)
        ac=a+s.position_gain_s2*(p-y[:2])+s.velocity_gain_s*(v-y[2:4])
        ac*=min(1.,s.acceleration_command_limit_m_s2/max(np.linalg.norm(ac),1e-12))
        F=s.striker_mass_kg*ac-drag(y[2:4],s.striker_drag_N,s.drag_smoothing_m_s)
        self.Ttarget=allocate(e.T,F,s.minimum_target_tension_N,s.maximum_target_tension_N,
                              s.allocation_iterations,s.allocation_regularization,self.Ttarget)
        self.force_residual=float(np.linalg.norm(e.T@self.Ttarget-F))
        # Positive winding omega = e.v/r. Differentiate cable geometry for alpha.
        omega_ff=e@v/self.r
        alpha_ff=(e@a-(float(v@v)-(e@v)**2)/L)/self.r
        self.command=(self.Ttarget*self.r+self.j*alpha_ff+
                      s.tension_feedback_gain*self.r*(self.Ttarget-T)+
                      s.reel_speed_gain_Nms*(omega_ff-y[12:16])+
                      self.b*y[12:16]+self.friction(T,y[12:16]))

    def wall_force(self,p,v,r,puck=False):
        s=self.s;F=np.zeros(2)
        in_goal=abs(p[0]-s.table_width_m/2)<s.goal_width_m/2-r
        for axis,bound in enumerate([s.table_width_m,s.table_length_m]):
            if puck and axis==1 and in_goal: continue
            overlap=r-p[axis]
            if overlap>0:F[axis]+=max(0.,s.wall_stiffness_N_m*overlap-s.wall_damping_Ns_m*v[axis])
            overlap=p[axis]+r-bound
            if overlap>0:F[axis]-=max(0.,s.wall_stiffness_N_m*overlap+s.wall_damping_Ns_m*v[axis])
        return F

    def derivative(self,y):
        s=self.s;T,_,e,*_=self.strings(y)
        contact_force,_,_=contact(y[:2],y[2:4],y[4:6],y[6:8],s.striker_radius_m,s.puck_radius_m,s.contact_stiffness_N_m,s.contact_damping_Ns_m) if not self.scored else (np.zeros(2),0,0)
        dy=np.zeros(20);dy[:2]=y[2:4]
        dy[2:4]=(e.T@T+drag(y[2:4],s.striker_drag_N,s.drag_smoothing_m_s)-contact_force+
                 self.wall_force(y[:2],y[2:4],s.striker_radius_m))/s.striker_mass_kg
        if not self.scored:
            dy[4:6]=y[6:8]
            dy[6:8]=(contact_force+drag(y[6:8],s.puck_drag_N,s.drag_smoothing_m_s)+
                     self.wall_force(y[4:6],y[6:8],s.puck_radius_m,True))/s.puck_mass_kg
        omega=y[12:16];limit=self.limits(omega)
        # Hard output envelope in addition to the first-order driver state.
        applied=np.clip(y[16:20],-limit,limit)
        dy[8:12]=omega
        dy[12:16]=(applied-T*self.r-self.b*omega-self.friction(T,omega))/self.j
        dy[16:20]=(np.clip(self.command,-limit,limit)-y[16:20])/self.response
        return dy


def simulate(s=None,motors=None,progress=None):
    s=s or Settings();motors=motors or default_motors();m=Model(s,motors)
    dt=s.integration_step_s;steps=int(math.ceil(s.duration_s/dt));period=round(s.control_period_s/dt)
    times=np.linspace(0,steps*dt,steps+1);state=np.zeros((steps+1,20))
    T=np.zeros((steps+1,4));targets=T.copy();torques=T.copy();limits=T.copy();cmds=T.copy()
    ref=np.zeros((steps+1,2));resid=np.zeros(steps+1);fn=np.zeros(steps+1);overlap=fn.copy();scored=np.zeros(steps+1,dtype=bool)
    y=m.initial.copy();hit_times=[];last_contact=False
    for n,t in enumerate(times):
        if n%period==0:m.control(t,y)
        state[n]=y;T[n]=m.strings(y)[0];targets[n]=m.Ttarget
        limits[n]=m.limits(y[12:16]);torques[n]=np.clip(y[16:20],-limits[n],limits[n]);cmds[n]=m.command
        ref[n]=m.reference(t)[0];resid[n]=m.force_residual;scored[n]=m.scored
        if not m.scored:
            _,fn[n],overlap[n]=contact(y[:2],y[2:4],y[4:6],y[6:8],s.striker_radius_m,s.puck_radius_m,s.contact_stiffness_N_m,s.contact_damping_Ns_m)
        if overlap[n]>0 and not last_contact: hit_times.append(float(t))
        last_contact=overlap[n]>0
        if n==steps:break
        # Classical RK4, sample-and-hold controller. Every substage recomputes
        # geometry, actual tension, motor envelope, drag, contact, and wall forces.
        k1=m.derivative(y);k2=m.derivative(y+.5*dt*k1)
        k3=m.derivative(y+.5*dt*k2);k4=m.derivative(y+dt*k3)
        y=y+dt*(k1+2*k2+2*k3+k4)/6
        if not np.all(np.isfinite(y)) or np.linalg.norm(y[2:4])>100 or np.linalg.norm(y[:2])>10:
            raise ValueError('Simulation diverged. Reduce timestep, stiffness or controller gains.')
        if np.any(m.l0-m.r*y[8:12]<=0):raise ValueError('A reel wound beyond its initial free cable length; adjust motion/controller.')
        if not m.scored and (y[5]>s.table_length_m+s.puck_radius_m or y[5]<-s.puck_radius_m):
            m.scored=True;m.goal='opponent' if y[5]>s.table_length_m else 'robot';y[6:8]=0.
        if progress and n%500==0:progress(n/steps)
    out={'settings':s,'motors':motors,'t':times,'state':state,'tension':T,'target_tension':targets,
         'torque':torques,'torque_limit':limits,'torque_command':cmds,'reference':ref,
         'allocation_residual':resid,'contact_force':fn,'overlap':overlap,'scored':scored,
         'hit_times':hit_times,'goal':m.goal,'l0':m.l0,'anchors':m.anchors}
    return out


def summary(d):
    y=d['state'];s=d['settings'];dt=s.integration_step_s
    return {'first_contact_s':d['hit_times'][0] if d['hit_times'] else None,
            'goal':d['goal'],'peak_tension_N_each':d['tension'].max(axis=0).tolist(),
            'peak_motor_rpm_each':(np.abs(y[:,12:16]).max(axis=0)*60/(2*np.pi)).tolist(),
            'max_striker_speed_m_s':float(np.linalg.norm(y[:,2:4],axis=1).max()),
            'max_puck_speed_m_s':float(np.linalg.norm(y[:,6:8],axis=1).max()),
            'peak_contact_force_N':float(d['contact_force'].max()),
            'max_contact_overlap_mm':float(d['overlap'].max()*1000),
            'max_tracking_error_mm':float(np.linalg.norm(y[:,:2]-d['reference'],axis=1).max()*1000),
            'max_force_allocation_residual_N':float(d['allocation_residual'].max()),
            'slack_time_s_each':((d['tension'][:-1]<=1e-6).sum(axis=0)*dt).tolist(),
            'above_continuous_torque_time_s_each':((np.abs(d['torque'][:-1])>np.array([m.continuous_torque_Nm for m in d['motors']])).sum(axis=0)*dt).tolist(),
            'torque_limited_time_s_each':((np.abs(d['torque_command'][:-1])>d['torque_limit'][:-1]+1e-9).sum(axis=0)*dt).tolist(),
            'integration_step_s':dt,'recorded_samples':len(d['t'])}


def export_csv(d,path):
    s=d['settings'];y=d['state'];r=np.array([m.reel_diameter_mm/2000 for m in d['motors']])
    length=np.linalg.norm(d['anchors'][None,:,:]-y[:,None,:2],axis=2)
    natural=d['l0'][None,:]-r*y[:,8:12]
    columns=[d['t'],*y[:,:8].T,*d['reference'].T,d['contact_force'],d['overlap'],d['allocation_residual'],d['scored'].astype(int)]
    names=['time_s','striker_x_m','striker_y_m','striker_vx_m_s','striker_vy_m_s','puck_x_m','puck_y_m','puck_vx_m_s','puck_vy_m_s',
           'target_x_m','target_y_m','contact_force_N','contact_overlap_m','allocation_residual_N','goal_scored']
    for i in range(4):
        for label,array in [('tension_N',d['tension']),('target_tension_N',d['target_tension']),
                            ('length_m',length),('unstretched_length_m',natural),('extension_m',length-natural),
                            ('rpm',y[:,12:16]*60/(2*np.pi)),('torque_Nm',d['torque']),
                            ('torque_command_Nm',d['torque_command']),('available_torque_Nm',d['torque_limit'])]:
            names.append(f'M{i+1}_{label}');columns.append(array[:,i])
    np.savetxt(path,np.column_stack(columns),delimiter=',',header=','.join(names),comments='',fmt='%.9g')
