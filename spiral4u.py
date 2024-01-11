# Clothoid 
# - Euler Spiral
# - Fresnel
# - Cornu


import numpy as np
import matplotlib.pyplot as plt

from scipy.special import fresnel


# reference:
#  1. https://pwayblog.com/2016/07/03/the-clothoid/
#  1. https://en.wikipedia.org/wiki/Euler_spiral
#
# clothoid - continuous, linear curvature variation
#
# Clothoid gemoetry and math
#
# -> A_sq = rl = Ri*Li = c*t
#
#  ; A - flatness / homothetic parameter
#
#  ; (x=0.,y=0.) at (R=Inf,L=0.0)
#  ; (x=0.5*A/sqrt(pi), y=0.5*A/sqrt(pi)) at (R=0,L=Inf)
#
#
# The angle ai ; measured beteween the tangent in the current point i and initial direction (R=Inf)
#
# -> ai = Li_sq / (2*A_sq)
#       = Li_sq / (2*R*L)
#       = Li / (2*Ri)
#
def is_positive(v):
    return True if v>0 else False

def azi2u(azi):
    return np.array([np.cos(azi), np.sin(azi)]).T





class Base:
    def __init__(self,s,x,y,hdg,length):
        self.s = s
        self.x = x
        self.y = y
        self.hdg = hdg
        self.length = length

class Line(Base):
    def __init__(self,s,x,y,hdg,length):
        super().__init__(s,x,y,hdg,length)

    def xy_at_ls(self,l):
        return np.array([[self.x, self.y]]) + [azi2u(self.hdg)]*np.array(l).reshape((-1,1))

    def hdg_at_ls(self,l):
        return np.array([self.hdg]*len(l))

class Spiral(Base):
    def __init__(self,s,x,y,hdg,length, curvStart,curvEnd):
        super().__init__(s,x,y,hdg,length)

        self.curvStart = curvStart
        self.curvEnd = curvEnd

        # euler spiral
        assert is_positive(curvStart)==is_positive(curvEnd)

        abs_c1, abs_c2 = abs(curvStart), abs(curvEnd)
        if abs_c1 < abs_c2:
            self._forward = True
            self._spiral = spiral = EulerSpiral.from_ccl(abs_c1, abs_c2, length)
            self._lmin = spiral.l_at_c(abs_c1)
            self._lmax = spiral.l_at_c(abs_c2)
        else:
            self._forward = False
            self._spiral = spiral = EulerSpiral.from_ccl(abs_c2, abs_c1, length)
            self._lmin = spiral.l_at_c(abs_c2)
            self._lmax = spiral.l_at_c(abs_c1)

        self._sign = 1.0 if is_positive(curvStart) else -1.0
        #
        # self.length = abs(self._lmin-self._lmax)

    @classmethod
    def from_ccq(cls,s,x,y,hdg,c1,c2,theta):
        c11 = abs(c1)
        c22 = abs(c2)
        if c11>c22:
            c11,c22 = c22,c11
        q = abs(theta)
        spiral = EulerSpiral.from_ccq(c11,c22,q)
        #
        l1 = spiral.l_at_c(c11)
        l2 = spiral.l_at_c(c22)
        length = l2-l1
        return cls(s,x,y,hdg,length,curvStart,curvEnd)

    def xy_at_ls(self, l):
        spiral = self._spiral
        ps_ = spiral.xy_at_l(self._lmin+l)
        
        #print(ps_)
        p00 = spiral.xy_at_l(self._lmin)
        p11 = spiral.xy_at_l(self._lmax)
        
        
        #p0 = ps_[0] if self._forward else ps_[-1]
        p0 = p00 if self._forward else p11
        ps0 = ps_-p0
        q_offset = 0.0
        if self._forward:
            if self._sign < 0 :
                ps0[:,1] = -ps0[:,1]
        else:
            q_offset = np.pi
            if self._sign > 0 :
                ps0[:,1] = -ps0[:,1]

        q0 = spiral.theta_at_l([self._lmin,self._lmax])[0 if self._forward else 1]
        if not self._forward:
            q0 = -q0

        q0 = q0+q_offset
        if self._sign<0:
            q0 = -q0
        q = -q0+self.hdg
        # q = -q0

        R = np.array([
                    [np.cos(q), -np.sin(q)],
                    [np.sin(q), np.cos(q)]])

        return np.array([[self.x, self.y]]) + ps0@R.T

    def hdg_at_ls(self,l):
        spiral = self._spiral
        qs_ = spiral.theta_at_l(self._lmin+l)
        
        q00 = spiral.theta_at_l(self._lmin)
        q11 = spiral.theta_at_l(self._lmax)

        
        q0 = q00 #qs_[0]
        dq = qs_ - q0
        if self._sign<0:
            dq = -dq
        
        return dq+self.hdg

class Arc(Base):
    def __init__(self,s,x,y,hdg,length, curvature):
        super().__init__(s,x,y,hdg,length)

        self.curvature = curvature

        self._pos_center = np.array([x,y]) + azi2u(hdg+np.pi/2)/curvature
        
        print('arc center', self._pos_center)
        

    def xy_at_ls(self,l):
        '''
        dq = l/r = l*c
        q = l*c + q0
        '''
        qs = self.hdg_at_ls(l)
        return [self._pos_center] + azi2u(qs-np.pi/2)/self.curvature 

    def hdg_at_ls(self,l):
        # print(l)
        # print(self.curvature)
        # print(l*self.curvature)
        return np.array(l)*self.curvature + self.hdg

class EulerSpiral:
    def __init__(self, RL):
        self.RL = RL
        return

    def __repr__(self):
        return f'EulerSpiral(RL={self.RL:.3f})'

    @classmethod
    def from_rl(cls,r,l):
        return cls(r*l)

    ##@classmethod
    ##def from_rrq(cls, r1,r2,theta):
    ##    '''
    ##    solve

    ##    q1 = l1/r1/2.0
    ##       = RL/(r1**2)/2.0
    ##    q2 = l2/r2/2.0
    ##       = RL/(r2**2)/2.0

    ##    theta = q2-q1 = RL/2.0 * ( 1/(r2**2) - 1/(r1**2) )

    ##    => RL = 2.0 * theta / ( 1/(r2**2) - 1/(r1**2) )
    ##    '''

    ##    RL = 2.0 * theta / ( 1/(r2**2) - 1/(r1**2) )

    ##    return cls(RL)

    @classmethod
    def from_ccq(cls, c1,c2,theta):
        '''
        solve

        q1 = l1/r1/2.0
           = RL/(r1**2)/2.0
        q2 = l2/r2/2.0
           = RL/(r2**2)/2.0

        theta = q2-q1 = RL/2.0 * ( 1/(r2**2) - 1/(r1**2) )

        => RL = 2.0 * theta / ( 1/(r2**2) - 1/(r1**2) )
        '''

        RL = 2.0 * theta / ( c2**2 - c1**2 )

        return cls(RL)

    @classmethod
    def from_ccl(cls, c1,c2,length):
        '''
        solve

        RL = l1*r1
        l1 = RL / r1 = RL * c1
        l2 = RL / r2 = RL * c2

        length = l2-l1 = RL(c2-c1)
        RL = length/(c2-c1)

        '''

        RL = length/(c2-c1)

        return cls(RL)

    def theta_at_l(self, l):
        '''
        RL = R_s*L_s = contant
        -> 1/R_s = L_s/RL

        theta_s = L_s / (2*R_s)
                = (L_s**2)/(2*RL)

        '''
        return (np.array(l)**2)/self.RL/2.0

    def xy_at_l(self, l):
        '''
        scipy fresnel formula is different from normalized euler spiral
            C(t1) = intgral(0,t1)(cos(pi/2 * t**2))
            S(t1) = intgral(0,t1)(sin(pi/2 * t**2))

        we want, (normalized euler spiral)
            NC(s1) = intgral(0,s1)(cos(s**2))
            NS(s1) = intgral(0,s1)(sin(s**2))

        ...
            substitute pi/2*(t**2) -> s**2

            -> s = t * sqrt(pi/2)
            -> t = s / sqrt(pi/2)

            dt = ds / sqrt(pi/2)

            => NC(s1) = sqrt(pi/2) * C(t1)

        '''
        # scale down to normalized Euler spiral
        SCALE = np.sqrt(2.0*self.RL)
        s = l/SCALE

        # get fresnel integral of normalized ...
        SQRT_PI_2 = np.sqrt(np.pi/2.0)
        t = s/SQRT_PI_2
        ns_,nc_ = fresnel(t)
        ncs = nc_*SQRT_PI_2
        nss = ns_*SQRT_PI_2

        # scale back
        cs = ncs*SCALE
        ss = nss*SCALE

        # RL = self.RL
        # x = l - (l**5)/(40*(RL**2)) + (l**9)/(3456*(RL**4)) - (l**13)/(599040*(RL**6))  # + ...
        # y = (l**3)/(6*RL) - (l**7)/(336*(RL**3)) + (l**11)/(42240*(RL**5)) - (l**15)/(9676800*(RL**7)) # + ...

        if type(l) == np.ndarray:
            return np.c_[cs,ss]
        else:
            return np.array([cs,ss])

    def l_at_r(self, r):
        return self.RL/r

    def r_at_l(self, l):
        return self.RL/l

    def l_at_c(self, c):
        return self.RL*c

    def c_at_l(self, l):
        return l/self.RL

    def rotation_center_at_l(self, l):
        p = self.xy_at_l(l)
        r = self.r_at_l(l)

        qv = self.theta_at_l(l)+np.pi/2.
        v = np.array([np.cos(qv), np.sin(qv)])

        if type(l) == np.ndarray:
            return p + v.T*r[:,None]
        else:
            return p + v*r


def test1() :

    ls = np.arange(0.,10.0,0.01)
    # print(ls)

    spiral = EulerSpiral(1.0)
    xy = spiral.xy_at_l(ls)


    plt.figure()
    plt.plot(xy[:,0],xy[:,1])

    plt.grid()
    # plt.legend()
    plt.axis('equal')

    plt.show()


def test2() :

    r1,r2,dq = 1500., 9., np.radians(20.62)

    spiral = EulerSpiral.from_ccq(1/r1,1/r2,dq)

    l1 = spiral.l_at_r(r1)
    l2 = spiral.l_at_r(r2)
    ls = np.arange(l1,l2,0.01)

    xy = spiral.xy_at_l(ls)

    cs = spiral.rotation_center_at_l(np.linspace(l1,l2,10))

    plt.figure()
    plt.plot(xy[:,0],xy[:,1])
    plt.plot(cs[:,0],cs[:,1],'m*-')

    plt.grid()
    # plt.legend()
    plt.axis('equal')

    plt.show()


def translate_rotate(ps, translation, heading):
    ps1 = ps + [translation]
    R = np.array([
                 [np.cos(heading), -np.sin(heading)],
                 [np.sin(heading), np.cos(heading)]])
    # return (R@ps1.T).T
    return ps1@R.T


def ncap_road() :

    r1,r2,dq = 1500., 9., np.radians(20.62)

    spiral = EulerSpiral.from_ccq(1/r1,1/r2,dq)

    l1 = spiral.l_at_r(r1)
    l2 = spiral.l_at_r(r2)
    q1 = spiral.theta_at_l(l1)
    q2 = spiral.theta_at_l(l2)

    xy = spiral.xy_at_l(np.linspace(l1,l2,20))
    pc = spiral.rotation_center_at_l(np.array([l2]))
    print('radius', np.linalg.norm(xy[-1]-pc))

    heading0 = 0.0
    heading1 = q2-q1
    print('heading', np.degrees([heading0, heading1]))
    print(20.62*2 + 48.76)
    print('distance', l2-l1)

    xy_cl = translate_rotate(xy, -xy[0], -heading0)
    rc_cl = translate_rotate(pc, -xy[0], -heading0)


    plt.figure()
    plt.plot(xy_cl[:,0],xy_cl[:,1])
    plt.plot(rc_cl[:,0],rc_cl[:,1],'m*-')
    plt.plot([xy_cl[-1,0],rc_cl[0,0]],[xy_cl[-1,1],rc_cl[0,1]],'r-')

    plt.grid()
    # plt.legend()
    plt.axis('equal')

    plt.show()


def test3():

    #<geometry s="0.0000000000000000e+00" x="1.1999999999998883e+02" y="1.3700000000000000e+02" hdg="-1.5707963267998624e+00" length="5.1868687057179841e-01">
    #    <line/>
    #</geometry>
    #<geometry s="5.1868687057179841e-01" x="1.1999999999998883e+02" y="1.3648131312942820e+02" hdg="-1.5707963268059659e+00" length="8.9414893617021267e+00">
    #    <spiral curvStart="-0.0000000000000000e+00" curvEnd="-8.5106382978723402e-02"/>
    #<geometry s="9.4601762322739251e+00" x="1.1887762726395758e+02" y="1.2766840661121266e+02" hdg="-1.9512852358133257e+00" length="9.5153674781379021e+00">
    #    <arc curvature="-8.5106382978723402e-02"/>
    #</geometry>

    c1,c2 = 0, 8.5106382978723402e-02
    dq = 1.9512852358133257e+00 - 1.5707963268059659e+00
    dl = 6.439328083202938

    spiral1 = EulerSpiral.from_ccq(c1,c2,dq)
    spiral2 = EulerSpiral.from_ccl(c1,c2,dl)

    print(spiral1)
    print(spiral2)

    


def test_geometry():

    plt.figure()

    s = 0.0
    x,y = -20.0, 0.0
    hdg = 0.0
    
    curvature = 1/9.0

    # line
    length = 20.0
    geom = Line(s,x,y,hdg,length)

    ls = np.linspace(0.,geom.length)
    ps = geom.xy_at_ls(ls)
    plt.plot(ps[:,0],ps[:,1])
    
    
    # spiral
    s=s+geom.length
    # print(geom.length)
    x, y = geom.xy_at_ls([geom.length])[0]
    hdg = geom.hdg_at_ls([geom.length])[0]
    length = 6.439328083202938
    curvStart,curvEnd = 1/1500. , curvature
    # curvStart,curvEnd = curvature, curvature/2
    # curvStart,curvEnd = -curvature, -1/1000.
    # curvStart,curvEnd = -curvature, -curvature/2
    geom = Spiral(s, x, y, hdg, length, curvStart,curvEnd)

    ls = np.linspace(0.,geom.length)
    ps = geom.xy_at_ls(ls)
    plt.plot(ps[:,0],ps[:,1])
    
           
    # arc
    s=s+geom.length
    x,y = geom.xy_at_ls(np.array([geom.length]))[0]
    
    
    
    hdg = geom.hdg_at_ls(np.array([geom.length]))[0]
    length = 7.6592
    curvature = 1/9.0
    geom = Arc(s, x, y, hdg, length, curvature)


    ls = np.linspace(0.,geom.length)
    ps = geom.xy_at_ls(ls)
    plt.plot(ps[:,0],ps[:,1])
    plt.plot(geom._pos_center[0],geom._pos_center[1],'m*-')


    # spiral
    s=s+geom.length
    # print(geom.length)
    x,y = geom.xy_at_ls([geom.length])[0]
    hdg = geom.hdg_at_ls([geom.length])[0]
    length = 6.439328083202938
    curvStart,curvEnd = curvature, 1/1500.
    # curvStart,curvEnd = curvature, curvature/2
    # curvStart,curvEnd = -curvature, -1/1000.
    # curvStart,curvEnd = -curvature, -curvature/2
    geom = Spiral(s, x, y, hdg, length, curvStart,curvEnd)
    #geom = Spiral.from_ccq(0.0, x, y, hdg, curvStart, curvEnd, delta_theta)
    
    


    ls = np.linspace(0.,geom.length)
    ps = geom.xy_at_ls(ls)
    plt.plot(ps[:,0],ps[:,1])



    plt.grid()
    plt.axis('equal')

    plt.show()
    


    return



if __name__ == '__main__':

    # test1()
#
    # test2()

    # ncap_road()

    # test3()

    test_geometry()





