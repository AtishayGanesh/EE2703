from pylab import *
import scipy.special as sp

def g(t,A,B):
	y=A*sp.jn(2,t)+B*t # f(t) vector
	return(y)

def g_matrix(t,A=1.05,B=-0.105):
	J_val = sp.jn(2,t)
	t_val = t
	M = c_[J_val,t_val]
	x = array([A,B])
	return(matmul(M,x),M)

def mse(data,assumed_model):
	return(sum(square(data-assumed_model))/101)

#Part 2
a = loadtxt("fitting.dat")
time = a[:,0]
values = a[:,1:]
true_fn = g(time,1.05,-0.105)
values = c_[values,true_fn]
#Part 3
figure(0)
scl=logspace(-1,-3,9) # noise stdev
plot(time,values)
title(r'Plot of the data')
legend(list(scl)+["True Value"])
grid(True)
show()

#Part 4

figure(0)
plot(time,true_fn)
xlabel(r"$t$",size =20)
ylabel(r"true value",size =20)
grid(True)

#Part 5
figure(1)
plot(time,c_[values[:,0],true_fn])
std = std(values[:,0]-true_fn)
errorbar(time[::5],values[:,0][::5],std,fmt='ro')
show()

#Part 6
figure(0)
supposed_true = g_matrix(time)[0]
plot(time,supposed_true)
title("Plot using Matrix Multiplication")
xlabel(r"$t$",size =20)
ylabel(r"true value",size =20)
legend(["True Value"])

#Part 7 &8
A_range = arange(0,2.1,0.1)
B_range = arange(-0.2,0.01,0.01)
figure(1)
e_matrix = zeros((len(A_range),len(B_range)))
for A in enumerate(A_range):
	for B in enumerate(B_range):
		e_matrix[A[0]][B[0]] = mse(values[:,0],g_matrix(time,A[1],B[1])[0])
contour_obj = contour(A_range,B_range,e_matrix,arange(0,20*0.025,0.025))
clabel(contour_obj,contour_obj.levels[0:5])
title("Contour Plot")
plot(1.05, -0.105,'ro', label = 'Exact Value')
annotate(s ="Exact Value",xy = [0.8,-0.100])

#Part 9,10
l_mse_A = zeros((9))
l_mse_B = zeros((9))
l_mse_error = zeros((9))
for i in range(9):
	temp = linalg.lstsq(g_matrix(time)[1],values[:,i],rcond=None)
	l_mse_error[i] = temp[1][0]
	l_mse_A[i],l_mse_B[i] = temp[0]
figure(2)
plot(scl,l_mse_error)
plot(scl,absolute(l_mse_A-1.05),'ro')
plot(scl,absolute(l_mse_B+0.105),'go')
title("Error vs Noise")

#part 11
figure(3)
loglog(scl,l_mse_error)
loglog(scl,absolute(l_mse_A-1.05),'ro')
loglog(scl,absolute(l_mse_B+0.105),'go')
title("Log Error vs Log Noise")
show()