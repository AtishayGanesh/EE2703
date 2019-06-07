import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import matplotlib
def plotter(fig_no,arg1,arg2,
    label_x,label_y,type=plt.plot,
    arg3='b-',title="",cmap = matplotlib.cm.jet):
    '''plotter function to help with standard plots'''
    plt.figure(fig_no)
    plt.grid()
    if type==plt.contourf:
        type(arg1,arg2,arg3,cmap=cmap)
        '''mostly only contour graphs use a colormap'''
        plt.colorbar()
    else:
        type(arg1,arg2,arg3)
    plt.xlabel(label_x,size =19)
    plt.ylabel(label_y,size =19)
    plt.title(title)

def bodeplot(w,s,phi):
    plt.subplot(2,1,1)
    plt.semilogx(w,s)
    plt.xlabel(r'$\omega$',size=17)
    plt.ylabel(r'$|H(j\omega)|$',size =17)
    plt.subplot(2,1,2)
    plt.semilogx(w,phi)
    plt.xlabel(r'$\omega$',size=17)
    plt.ylabel(r'$\angle(H(j\omega))$',size =17)

def input_fn(decay=0.5,cos_term=1.5):
    '''input function to the single spring system'''
    return (np.poly1d([1,decay]),
    np.poly1d([1,2*decay,cos_term**2+decay**2]))

def transfer_fn():
    '''transfer function of the single spring system'''
    return (np.poly1d([1]), np.poly1d([1,0,2.25]))

def input_td(t,freq,decay):
    '''time domain expression of a generalised
     input to the spring system'''
    return(np.cos(freq*t)*np.exp(-1*decay*t))

def solve(decay):
    '''using the convolution property of the laplace 
    transform to solve for the output of the LTI system'''
    Hn, Hd = transfer_fn()
    Fn, Fd = input_fn(decay)
    np.polymul(Fn,Hn)
    t = np.linspace(0,50,200)
    Y = sp.lti(np.polymul(Fn,Hn),np.polymul(Fd,Hd))
    t, y = sp.impulse(Y,None,t)
    plotter(1,t,y,"t","x(t)",
        title="System response with decay {}".format(decay))
    plt.show()

#High decay rate
solve(0.5)
#Low decay rate
solve(0.05)

def loop_freq(decay = 0.05):
    '''This function shows the plot  at various frequencies
    and also calculates the bode plot parameters'''
    t = np.linspace(0,100,300)
    Hn,Hd = transfer_fn()
    H = sp.lti(Hn,Hd)
    color_list=['k','g','r','c','m']
    # Short forms for colors
    # Black Green Red Cyan and Magenta
    fl = np.arange(1.4,1.6,0.05)
    
    for i in range(5):
        u = input_td(t,fl[i],decay)
        t,y,svec = sp.lsim(H,u,t)
        plotter(1,t,y,"t","x(t)",
                title="System response with decay 0.05 and various frequencies",
                arg3=color_list[i]+'-')
    plt.legend(["freq {}".format(i) for i in fl])
    plt.show()
    return(H.bode())


w,s,phi = loop_freq()
#Making a bode plot to better understand 
#the variation of the response with the input frequency.
bodeplot(w,s,phi)
plt.title('Bode plot of Transfer function of spring system')
plt.show()

def coupled():
    '''solving the coupled springs problem'''
    t = np.linspace(0,20,200)
    #Transfer function for X
    X = sp.lti([1,0,2],[1,0,3,0])
    #Transfer function for Y
    Y = sp.lti([2],[1,0,3,0])
    t,x = sp.impulse(X, None, t)
    t,y = sp.impulse(Y, None, t)
    plotter(1,t,x,"t","f(t)",title="Coupled Spring system, plot of x")
    plotter(1,t,y,"t","f(t)",title="Coupled Spring system, plot of y",arg3 ="g-")
    plt.legend(["f(t) = x(t)","f(t) = y(t)"])
coupled()
plt.show()


def rlc_tf():
    '''Transfer function for the RLC Circuit'''
    return(np.poly1d([1]),np.poly1d([1e-12,1e-4,1]))

def rlc_input(t):
    '''Input to the RLC Circuit'''
    return(np.cos(1e3*t)-np.cos(1e6*t))

def solve_rlc():
    '''RLC Solver'''
    t = np.arange(0,10e-3,1e-7)
    Hn,Hd = rlc_tf()
    H = sp.lti(Hn,Hd)
    #Plotting the Bode plot for the RLC Circuit
    w,s,phi = H.bode()
    bodeplot(w,s,phi)
    plt.title('Bode plot of Transfer function of RLC Filter')
    plt.show()
    #Solving the RLC Circuit
    u = rlc_input(t)
    t,x,svec = sp.lsim(H,u,t)
    
    plt.rcParams.update({'mathtext.default':  'regular' })
    plotter(1,t,x,"t","V(t)",title="RLC Circuit, plot of output Voltage, Slow Time")
    plotter(2,t[0:300],x[0:300],"t","V(t)",title="RLC Circuit, plot of output Voltage, Fast Time")

    plt.ylabel('$V_{o}(t)$')
    plt.show()
solve_rlc()