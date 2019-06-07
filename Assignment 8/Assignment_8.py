#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pylab import *

x=rand(100)
X=fft(x)
y=ifft(X)
c_[x,y]
print(abs(x-y).max())

#4.446383891599849e-16
# In[2]:


x=linspace(0,2*pi,128)
y=sin(5*x)
Y=fft(y)
figure()
subplot(2,1,1)
plot(abs(Y),lw=2) #line width of 2
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)#for the grid
subplot(2,1,2)
plot(unwrap(angle(Y)),lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$k$",size=16)
grid(True)
savefig("fig8-1.png")#Automatically saving the figure
show()


# In[3]:


x=linspace(0,2*pi,129);x=x[:-1]#so that last point is excluded
y=sin(5*x)
Y=fftshift(fft(y))/128.0#fftshift converts from [0,2pi] to [-pi,pi] 
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)#highlighting points for which phase is relavant
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$k$",size=16)
grid(True)
savefig("fig8-2.png")
show()


# In[4]:


t=linspace(0,2*pi,129);t=t[:-1]#low sampling frequency,aliasing
y=(1+0.1*cos(t))*cos(10*t)#AM with carrier at 10 and modulating freq of 1
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig8-3.png")
show()


# In[5]:


t=linspace(-4*pi,4*pi,513);t=t[:-1]#higher sampling rate, no aliasing
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-15,15])
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
savefig("fig8-4.png")
show()


# In[96]:


def dft(func,tim= None,n_p=512,fig_no=0,name= None):
    '''Function to plot the dft using fft algorithm
func: lambda/function whose fourier transform is required
time: time interval in the form (start_time,end_time)
default is (-4*pi,4*pi)
n_p: number of samples in time domain
fig_no: Figure number
name: Name of function for the graph
'''
    if tim is None:
        t=linspace(-4*pi,4*pi,n_p,endpoint= False)
    else:
        start,end = tim
        t = linspace(start,end,n_p,endpoint=False)
    y = func(t)
    Y=fftshift(fft((y)))/n_p#fftshift so the plot is in terms we know
    w=linspace(-pi,pi,n_p,endpoint= False);
    w = w*n_p/(end-start)#the range of frequencies
    fig, (ax1, ax2) = plt.subplots(2, 1)
    Ysig = where(abs(Y)>10**-5)#only plot significant points phase
    ax1.plot(w,abs(Y),lw=1)
    ax1.set_xlim([-2*max(w[Ysig]),2*max(w[Ysig])])
    #to ensure that we dont go too far on each side

    ax1.set_ylabel(r"$|Y|$",size=16)
    title("Spectrum of {}".format(name))
    ax1.grid(True)
    ax2.plot(w[Ysig],angle(Y[Ysig]),'ro')
    ax2.set_xlim([-2*max(w[Ysig]),2*max(w[Ysig])])

    ax2.set_ylabel(r"Phase of $Y$",size=16)
    ax2.set_xlabel(r"$\omega$",size=16)
    grid(True)
    return ax1,ax2,w

y1 = lambda t : (cos(t))**3
y2 = lambda t : (sin(t))**3
y3 = lambda t : cos(20*t+5*cos(t))

dft(y1,(-4*pi,4*pi),256,1,r'$cos^{3}t$')
dft(y2,(-2*pi,2*pi),256,2,r"$sin^{3}t$")
dft(y3,(-4*pi,4*pi),2048,3,r"cos(20t+5cos(t))")
plt.show()


# In[135]:


def estctft(func,truth,tim= None,n_p=512,fig_no=0,name= None,):
    '''Function to plot the dft using fft algorithm
func: lambda/function whose fourier transform is desired
truth: analytic ctft function/lambda
tim: time interval in the form (start_time,end_time)
default is (-4*pi,4*pi)
n_p: number of samples in time domain
fig_no: Figure number
name: Name of function for the graph
'''
    if tim is None:
        start=-4*pi
        end=4*pi
    else:
        start,end = tim

    t = linspace(end,start,n_p,endpoint=False)
    y = func(t)
    #ifftshift needed to remove certain phase issues
    Y=fftshift(fft(ifftshift(y)))*(end-start)/(2*pi*n_p)

    w=linspace(-pi,pi,n_p,endpoint= False);
    w = w*n_p/(end-start)
    
    #sum of total difference between the two transforms
    error = sum(abs(truth(w)-Y))
    print(error)
    #2.2822972865535313e-14
    fig, (ax1, ax2) = plt.subplots(2, 1)
    Ysig = where(abs(Y)>10**-5)
    ax1.plot(w,abs(Y),lw=1)
    ax1.set_xlim([-2*max(w[Ysig]),2*max(w[Ysig])])

    ax1.set_ylabel(r"$|Y|$",size=16)
    title("Spectrum of {}".format(name))
    ax1.grid(True)
    ax2.plot(w[Ysig],angle(Y[Ysig]),'ro')
    ax2.set_xlim([-2*max(w[Ysig]),2*max(w[Ysig])])
    #Only plot significant phase points
    ax2.set_ylabel(r"Phase of $Y$",size=16)
    ax2.set_xlabel(r"$\omega$",size=16)
    grid(True)
    return ax1,ax2,w

y4 = lambda t : exp(t**2/-2)
y5 = lambda w : exp(-w**2/2)/sqrt(2*pi)
start = 4*pi
#plot the true CTFT gotten analytically
ax1,ax2,w = estctft(y4,y5,(-start,start),1024,4,r"$exp(-t^{2}/2)$")

ax1.plot(w,abs(y5(w)),'y-')
ax2.plot(w,angle(y5(w)),'yo')
show()