"""Run: python -m unittest -v test_physics.py. Physics checks, independent of GUI."""
import unittest
from dataclasses import replace
import numpy as np
from physics import *

class PhysicsTests(unittest.TestCase):
    def test_pretension_and_slack(self):
        m=Model(Settings(),default_motors());y=m.initial.copy()
        T,L,e,l,x,xd=m.strings(y)
        np.testing.assert_allclose(T,m.Ttarget,rtol=1e-10,atol=1e-10)
        self.assertLess(np.linalg.norm(e.T@T),.02)
        y[8:12]-=1
        np.testing.assert_array_equal(m.strings(y)[0],np.zeros(4))

    def test_contact_is_equal_opposite_and_dissipative(self):
        ms,mp=.2,.04;rs,rp=.045,.032;dt=1e-5
        z=np.array([0.,0.,0.,2.,0.,.09,0.,0.]);p0=ms*z[2:4]+mp*z[6:8];E0=.5*ms*np.dot(z[2:4],z[2:4])
        def f(z):
            F,_,_=contact(z[:2],z[2:4],z[4:6],z[6:8],rs,rp,20000.,6.)
            return np.r_[z[2:4],-F/ms,z[6:8],F/mp]
        for _ in range(3000):
            a=f(z);b=f(z+dt*a/2);c=f(z+dt*b/2);d=f(z+dt*c);z+=dt*(a+2*b+2*c+d)/6
        np.testing.assert_allclose(ms*z[2:4]+mp*z[6:8],p0,atol=1e-10)
        self.assertGreater(z[7],z[3]);self.assertLess(.5*ms*np.dot(z[2:4],z[2:4])+.5*mp*np.dot(z[6:8],z[6:8]),E0)

    def test_efficiency_loss_is_passive(self):
        m=Model(Settings(),default_motors());w=np.array([-50.,-1.,1.,50.])
        self.assertTrue(np.all(m.friction(np.ones(4)*10,w)*w>=0))
        m.eta[:]=1;np.testing.assert_allclose(m.friction(np.ones(4)*10,w),0)

    def test_motor_limits(self):
        m=Model(Settings(),default_motors());limit=m.limits(np.array([0,100,200,1000]))
        self.assertGreater(limit[0],limit[1]);self.assertEqual(limit[3],0)
        y=m.initial.copy();y[16:20]=100;dy=m.derivative(y);T=m.strings(y)[0]
        np.testing.assert_allclose(dy[12:16],(m.limits(np.zeros(4))-T*m.r)/m.j)

    def test_geometry_derivative(self):
        m=Model(Settings(),default_motors());p=np.array([.3,.35]);v=np.array([.1,1.2]);a=np.array([.2,4.]);h=1e-4;L,e=m.geometry(p)
        Lp=m.geometry(p+h*v+.5*h*h*a)[0];Lm=m.geometry(p-h*v+.5*h*h*a)[0]
        np.testing.assert_allclose((Lp-Lm)/(2*h),-e@v,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose((Lp-2*L+Lm)/(h*h),-e@a+(v@v-(e@v)**2)/L,rtol=1e-5,atol=1e-5)

    def test_full_run_and_timestep_convergence(self):
        s=replace(Settings(),duration_s=.60);d=simulate(s);fine=simulate(replace(s,integration_step_s=s.integration_step_s/2))
        self.assertTrue(d['hit_times']);self.assertTrue(np.all(d['tension']>=0))
        np.testing.assert_allclose(d['tension'][:,0],d['tension'][:,1],atol=1e-8)
        np.testing.assert_allclose(d['state'][-1,:8],fine['state'][-1,:8],atol=.004)
        np.testing.assert_allclose(d['tension'].max(axis=0),fine['tension'].max(axis=0),rtol=.03)
        self.assertTrue(np.all(np.abs(d['torque'])<=d['torque_limit']+1e-12))

    def test_weaker_motor_changes_motion(self):
        s=replace(Settings(),duration_s=.55);normal=simulate(s)
        weak=[replace(m,peak_torque_Nm=.10,continuous_torque_Nm=.05) for m in default_motors()];slow=simulate(s,weak)
        self.assertLess(np.linalg.norm(slow['state'][:,2:4],axis=1).max(),np.linalg.norm(normal['state'][:,2:4],axis=1).max())
        self.assertGreater(np.linalg.norm(slow['state'][-1,:2]-slow['reference'][-1]),.03)

if __name__=='__main__':unittest.main()
