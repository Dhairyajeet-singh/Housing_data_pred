import matplotlib.pyplot as plt
import numpy as np


m=[1,2,3,4,5]
y1=[100,50,30,80,150]
y2=[110,130,65,70,40]
y3=[150,90,40,60,70]
y4=[80,30,65,38,101]
plt.figure(figsize=(10,8))

plt.bar(m,y1,color='black',label='royal enfield',width=0.4)
plt.bar(m,y2,color='red',label='ninja',width=0.1)
plt.bar(m,y3, color='green',label='honda',width=0.2)
plt.bar(m,y4, color='yellow',label='harley',width=0.3)
plt.title('data representation of bikes using bar', fontsize='20')
plt.xlabel('number of days',fontsize=15)
plt.ylabel('distance in km',fontsize=15)
plt.xticks(np.arange(1,6,step=1))
plt.yticks(np.arange(0,201,step=20))
plt.legend()
plt.show()