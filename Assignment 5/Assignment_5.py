import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import lstsq
import sys

def plotter(fig_no,arg1,arg2,label_x,label_y,type=plt.plot,arg3='b-',title="",cmap = matplotlib.cm.jet):
	'''Function to help make plots'''
    plt.figure(fig_no)
    plt.grid()
    if type==plt.contourf:
        type(arg1,arg2,arg3,cmap=cmap)
        plt.colorbar()
    else:
        type(arg1,arg2,arg3)

    plt.xlabel(label_x,size =19)
    plt.ylabel(label_y,size =19)
    plt.title(title)
    

Nx = 25
Ny = 25
Niter = 1500
if len(sys.argv)>1:
    try:
        Ny = int(sys.argv[1])
        Nx = int(sys.argv[2])
        Niter = int(sys.argv[3])  
    except:
    	#incase the args are not input take the default values.
        pass
phi = np.zeros((Ny,Nx))
y = np.linspace(-0.5,0.5,Ny)
x = np.linspace(-0.5,0.5,Nx)
Y,X = np.meshgrid(y,x)
ii = np.where(X*X+Y*Y<=0.35*0.35) #find which points are close enough to count.
phi[ii]=1.0
plotter(1,x,y,"X axis","Y axis",plt.contourf,phi,"Contour plot of potential",cmap=matplotlib.cm.hot)
plt.plot(ii[0]/Nx-0.48,ii[1]/Ny-0.48,'ro') #We need to shift the ii so that it will be plotted at the centre of the 1*1 region.
plt.show()

errors=np.zeros(Niter)
for i in range(Niter):
    oldphi=phi.copy()#Need a proper copy not a view
    phi[1:-1,1:-1]=0.25*(phi[1:-1,0:-2]+ phi[1:-1,2:]+
        phi[0:-2,1:-1]+ phi[2:,1:-1]);#Laplace Equation
    phi[1:-1,0]=phi[1:-1,1]
    phi[1:-1,-1]=phi[1:-1,-2]
    phi[-1,:]=phi[-2,:]
    phi[ii]=1.0
    #Boundary condition
    errors[i]=(abs(phi-oldphi)).max();
    #Calculate errors.


plotter(1,np.arange(Niter),errors,"Iteration number","Error",type=plt.semilogy,title='Error versus iteration number')
plotter(2,np.arange(500,Niter),errors[500:],"Iteration number","Error",type=plt.semilogy,title='Error versus iteration number above 500')
plotter(3,np.arange(Niter),errors,"Iteration number","Error",type=plt.loglog,title='Error versus iteration number loglog')
plt.show()

lst =lstsq(np.c_[np.ones(Niter-500),np.arange(Niter-500)],np.log(errors[500:]))#Lstsq for >500 iteration number
a,b =np.exp(lst[0][0]),lst[0][1] 
print(a,b)
plotter(1,np.arange(500,Niter),a*np.exp(b*np.arange(Niter-500)),"Iteration number","error",type=plt.semilogy,arg3="r-")
plotter(1,np.arange(500,Niter),errors[500:],"Iteration number","Error",type=plt.semilogy,title='Expected vs actual error (>500 iter)')
plt.legend(("Estimated Exponential","Actual Exponential"))
lstapprox =lstsq(np.c_[np.ones(Niter),np.arange(Niter)],np.log(errors))#lstsq for all iterations.
a,b = np.exp(lstapprox[0][0]),lstapprox[0][1]
print(a,b)
plotter(2,np.arange(Niter),a*np.exp(b*np.arange(Niter)),"Iteration number","Error",type=plt.semilogy,title='Error versus iteration number',arg3 = 'r-')
plotter(2,np.arange(Niter),errors,"Iteration number","Error",type=plt.semilogy,title='Expected vs actual')
plt.legend(("Estimated Exponential","Actual Exponential"))
plt.show()

#Potential Exponential and Contour Plot
fig4=plt.figure(4)
ax=p3.Axes3D(fig4) 
plt.title('3-D surface plot of potential')
surf = ax.plot_surface(Y, X, phi, rstride=1, cstride=1, cmap=matplotlib.cm.hot,linewidth=0)
fig4.colorbar(surf)
plt.show()
plotter(3,x,y,"X axis","Y axis",plt.contourf,phi,"Contour plot of potential",cmap=matplotlib.cm.hot)
plt.plot(ii[0]/Nx-0.48,ii[1]/Ny-0.48,'ro')

Jx = np.zeros((Ny,Nx))
Jy = np.zeros((Ny,Nx))
#Since J is vector we have Jx and Jy
Jx[:,1:-1] = 0.5*(phi[:,0:-2]-phi[:,2:])
Jy[1:-1,:] = 0.5*(phi[0:-2,:]-phi[2:,:])
plt.show()
plt.figure(0)
plt.quiver(x,y,Jx,Jy)
plt.plot(ii[0]/Nx-0.48,ii[1]/Ny-0.48,'ro')
plt.xlabel("X Axis",size = 19)
plt.ylabel("Y Axis",size=19)
plt.title("Current in the region")
plt.show()
phij = np.ones((Nx,Ny))*300
for i in range(Niter*2):
    phij[1:-1,1:-1]=0.25*(phij[1:-1,0:-2]+ phij[1:-1,2:]+
        phij[0:-2,1:-1]+ phij[2:,1:-1] + Jx[1:-1,1:-1]*Jx[1:-1,1:-1]+Jy[1:-1,1:-1]*Jy[1:-1,1:-1]);#Poisson's equation.
    phij[1:-1,0]=phij[1:-1,1]
    phij[1:-1,-1]=phij[1:-1,-2]
    phij[-1,:]=phij[-2,:]
    phij[0,:] = 300
    phij[ii] =300
    #Boundary condition.
plotter(1,x,y,"X axis","Y axis",plt.contourf,phij,"Contour plot of Temperature",cmap=matplotlib.cm.hot)
plt.show()
print(phij)