import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


delta = 0.01
a = -1
b = 1
n = 1000 # num of grid points

# x grid of n points
X, dx = np.linspace(a,b,n,retstep=True)
dt = 1
cur=0.4#пусть в нашей схеме r=0.4
c=cur*dx/dt
T0=700
T=500
# нач.усл.
def initial_u(y):
	if y<=0:
		return 1.
	else:
		return np.exp(-np.power((y/delta), 2))

# двумерный  массив u(t,x_j
U = np.zeros((T0,len(X)))

def u(x, t):#оно это делает для любого времени
	uvals = np.zeros(len(x)) # u values for this time step
	if t == 0: # начальное условие
		return np.array([initial_u(x[i]) for i in range(len(x))])
	else:
		for j in range(len(x)-1):
			if (j!=0) and (j!=len(x)):
				uvals[j] = U[t-1][j] + cur*cur*(U[t-1][j+1]-2*U[t-1][j] + U[t-1][j-1])/2 - cur*(U[t-1][j+1] - U[t-1][j-1])/2
			if j == 0: # левое гран условие
				uvals[j] = 1.
			if j == len(x): # левое гран условие
				uvals[j] = 0.
		return uvals

# solve for 700 time steps
for t in range(T0):
    U[t,:] = u(X,t)
    
U1 = np.zeros((len(X),T0))

for t in range (T0):
	for i in range(len(X)):
		U1[i,t] = initial_u(X[i]-c*t)#(1/cur+1)*


e_max = max(abs(U[T]-U1[:,T]))
R=np.power(abs(U[T]-U1[:,T]),2)
s = 0
for i in range(len(X)):
	s+=R[i]
e_max_q = math.sqrt(s/len(X))
print('Для t=',T,'макс. погрешность',e_max)
print('Для t=',T,'ср.квадр. погрешность',e_max_q)

# считаем полную вариацию
TV=0
E_TV = np.zeros(len(X))
E_TV = U[T]
for i in range(len(X)-1):
	TV+=abs(E_TV[i+1]-E_TV[i])
	
print('TV=',TV)

# plot solution

plt.figure(figsize=(6, 4))
plt.plot(X, U[:][T])
plt.plot(X, U1[:,T])

plt.show()

plt.style.use('dark_background')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# animate the time data
k = 0
def animate(i):
    global k
    x = U[k]
    y = U1[:,k]
    r=abs(U[k]-U1[:,k])
    k += 1
 
    ax1.clear()
    plt.plot(X,x,color='cyan')
    plt.plot(X,y,color='red')
    plt.plot(X,r,color='yellow')
    plt.xlabel(k-1)
    plt.grid(True)
    plt.ylim([-0.5,1.2])
    plt.xlim([-1,1])
    
    

anim = animation.FuncAnimation(fig,animate,frames=360,interval=20)
plt.show()
