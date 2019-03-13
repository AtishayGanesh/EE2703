import numpy  as np
import matplotlib.pyplot as plt
import scipy.integrate

def exp(x):
    '''Function to Calculate e^x'''
    return np.exp(x)

def coscos(x):
    '''Function to Calculate cos(cos(x)'''
    return np.cos(np.cos(x))

def plotter(fig_no,plot_x,plot_y,label_x,label_y,type=None,kind='b-',title=""):
    '''Function to Plot graphs. Grid is always present.
    Type defines the whether the graph is linear, loglog etc. Kind is the kind of symbols used.'''
    plt.figure(fig_no)
    plt.grid(True)
    if type =="semilogy":
        plt.semilogy(plot_x,plot_y,kind)
    elif type =='ll':
        plt.loglog(plot_x,plot_y,kind)
    elif type ==None:
        plt.plot(plot_x,plot_y,kind)
    plt.xlabel(label_x,size =19)
    plt.ylabel(label_y,size =19)
    plt.title(title)
    

def u1(x,k):
    '''coscos(x)*cos(kx)'''
    return(coscos(x)*np.cos(k*x))

def v1(x,k):
    '''coscos(x)*sin(kx)'''
    return(coscos(x)*np.sin(k*x))

def u2(x,k):
    '''exp(x)*cos(kx)'''
    return(exp(x)*np.cos(k*x))

def v2(x,k):
    '''exp(x)*sin(kx)'''
    return(exp(x)*np.sin(k*x))


def integrate():
    '''This function does the integration term by term in a single for loop. Uses quad integrator.'''
    a = np.zeros(51)
    b = np.zeros(51)
    a[0] =  scipy.integrate.quad(exp,0,2*np.pi)[0]/(2*np.pi)
    b[0] =  scipy.integrate.quad(coscos,0,2*np.pi)[0]/(2*np.pi)
    for i in range(1,51,2):
        a[i] = scipy.integrate.quad(u2,0,2*np.pi,args=(i//2+1))[0]/(np.pi)
        b[i] = scipy.integrate.quad(u1,0,2*np.pi,args=(i//2+1))[0]/(np.pi)
        a[i+1] = scipy.integrate.quad(v2,0,2*np.pi,args=(i//2+1))[0]/(np.pi)
        b[i+1] = scipy.integrate.quad(v1,0,2*np.pi,args=(i//2+1))[0]/(np.pi)
    return a,b
#Part 1
#1200 points since we do lstsq over 400 points later, hence it works out nicely.
t = np.linspace(-2*np.pi,4*np.pi,1200)
fr_length = np.arange(1,52)#Fourier Coefficient Number
#Now we plot the true functions
plotter(1,t,exp(t),r"$t$",r"exp(t)","semilogy",title ="Exponential Function on a semilog plot")
plotter(2,t,coscos(t),r"$t$",r"cos(cos(t))",title="Cos(Cos()) Function on a linear plot")
#now we plot the periodic extension of the function
plotter(1,t,np.concatenate((exp(t)[400:800],exp(t)[400:800],exp(t)[400:800])),r"$t$",r"exp(t)","semilogy",'r-')
plotter(2,t,np.concatenate((coscos(t)[400:800],coscos(t)[400:800],coscos(t)[400:800])),r"$t$",r"cos(cos(t)","semilogy",'r-')

#part 2
#We call the integrate function to get the fourier series coefficients in the right order
frexp,frcos = integrate()

#part 3
#we plot these in semilogy and loglog plots
plotter(3,fr_length,np.absolute(frexp),"Coefficient Number","Coefficient Value","semilogy",'ro',title="Semilog Fourier Coefficients for exp(t)")
plotter(4,fr_length,np.absolute(frexp),"Coefficient Number","Coefficient Value","ll",'ro',title="Log-Log Fourier Coefficients for exp(t)")
plotter(5,fr_length,np.absolute(frcos),"Coefficient Number","Coefficient Value","semilogy",'ro',title="Semilog Fourier Coefficients for coscos(t)")
plotter(6,fr_length,np.absolute(frcos),"Coefficient Number","Coefficient Value","ll",'ro',title="Loglog Fourier Coefficients for coscos(t)")


#parts 4 & 5
#using a least squares approach to the problem
x =np.linspace(0,2*np.pi,400,endpoint =True)
#the reason for endpoint=True is explained in the report
#b is the rhs of the equation
bexp = exp(x)
bcoscos =coscos(x)
#A is the matrix of values, made using 1 for loop.
A = np.zeros((400,51))
A[:,0] =1
for k in range(1,26):
    A[:,2*k-1] = np.cos(k*x)
    A[:,2*k] = np.sin(k*x)
#using lstsq to solve this function
cexp = np.linalg.lstsq(A,bexp)[0]
ccoscos = np.linalg.lstsq(A,bcoscos)[0]
#plotting these in the same graphs as above
plotter(3,fr_length,np.abs(cexp),"Coefficient Number","Coefficient Value","semilogy",'go',title="Semilog Fourier Coefficients for exp(t)")
plt.legend(("true","predicted"))
plotter(4,fr_length,np.abs(cexp),"Coefficient Number","Coefficient Value","ll",'go',title="Loglog Fourier Coefficients for exp(t)")
plt.legend(("true","predicted"))
plotter(5,fr_length,np.abs(ccoscos),"Coefficient Number","Coefficient Value","semilogy",'go',title="Semilog Fourier Coefficients for coscos(t)")
plt.legend(("true","predicted"))
plotter(6,fr_length,np.abs(ccoscos),"Coefficient Number","Coefficient Value","ll",'go',title="Loglog Fourier Coefficients for coscos(t)")
plt.legend(("true","predicted"))
#part 6
#we find the difference between the two vectors and take abs value, to find the max deviation.

diffexp = np.absolute(cexp-frexp)
diffcos = np.absolute(ccoscos-frcos)
print(np.amax(diffexp),np.amax(diffcos),diffexp,diffcos)

#part 7
# @ is a shortcut for matrix multiplication between numpy matrices.
Acexp = A@cexp
Accos = A@ccoscos

#we now plot these.
plotter(1,t,np.concatenate((np.zeros(400),Acexp,np.zeros(400))),r"$t$",r"exp(t)","semilogy",'go')
plt.legend(("true","periodic extension","predicted"))
plotter(2,t,np.concatenate((np.zeros(400),Accos,np.zeros(400))),r"$t$",r"coscos(t)",None,'go')
plt.legend(("true","periodic extension","predicted"))
plt.show()
