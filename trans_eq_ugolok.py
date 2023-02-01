import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


delta = 0.01
a = -1
b = 1
n = 1000 

# отрезок поделим на n точек
X, dx = np.linspace(a,b,n,retstep=True)
dt = 1
T0=700 #временной промежуток
T=500 #срез по времени, который нас интересует
cur=0.4#число Курранта 
c=cur*dx/dt

# начальное условие
def initial_u(y):
	if y<=0:
		return 1
	else:
		return np.exp(-np.power((y/delta), 2))

def u(x, t):#получаем численное решение на временном шаге t
    if t == 0: # начальное условие
        return np.array([initial_u(x[i]) for i in range(len(x))])
    uvals = [] # u values for this time step
    for j in range(len(x)):
        if j == 0: # левое граничное условие
            uvals.append(1.)
        else:
            uvals.append(U[t-1][j] - cur*(U[t-1][j]-U[t-1][j-1]))#схема явный левый уголок
    return uvals

# двумерный массив u(t,x_j) 
U = []
for t in range(T0):
    U.append(u(X, t))
    
U1 = np.zeros((len(X),T0))#знаем точное решение задачи
for t in range (T0):
	for i in range(len(X)):
		U1[i,t] = initial_u(X[i]-c*t)


e_max = max(abs(U[T]-U1[:,T]))#максимальное отклонение

R=np.power(abs(U[T]-U1[:,T]),2)
s = 0
for i in range(len(X)):
	s+=R[i]
e_max_q = math.sqrt(s/len(X))#среднеквадратичное отклонение

print('Для t=',T,'макс. погрешность',e_max)
print('Для t=',T,'ср.квадр. погрешность',e_max_q)

# считаем полную вариацию
TV=0
E_TV = np.zeros(len(X))
E_TV = U[T]
for i in range(len(X)-1):
	TV+=abs(E_TV[i+1]-E_TV[i])
	
print('Для t=',T, 'TV=',TV)


plt.figure(figsize=(6, 4))
plt.plot(X, U[:][T],color='green')
plt.plot(X, U1[:,T],color='red')

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
