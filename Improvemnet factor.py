
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[20]:


'''Improvement factor'''

def Improvement(x):
    
    Alpha = []
    
    CLB_new_data = np.loadtxt(str(x)+"_CLB.txt")
    CLB_old_data = np.loadtxt(str(x-1)+"_CLB.txt")
    NLPP_new_data = np.loadtxt(str(x)+"_NLPP.txt")
    NLPP_old_data = np.loadtxt(str(x-1)+"_NLPP.txt")

    for i in range(len(NLPP_new_data)):
        a = (NLPP_new_data[i,1])/(NLPP_old_data[i,1])
        print(a)
        Alpha.append(a)

    plt.plot(NLPP_new_data[:,0],Alpha[:])
    plt.title('Improvement Factor for Iteration Number'+str(x))
    plt.xlabel('l')
    plt.ylabel(r"$\frac{N^{\phi \phi}_{new} + C^{B B}_{new}}{N^{\phi \phi}_{old} + C^{B B}_{old}}$",fontsize = 18)


# In[22]:


Improvement(1)


# In[12]:


def Plot_Iteration(x):
    
    
    CLB_data = np.loadtxt(str(x)+"_CLB.txt")
    NLPP_data = np.loadtxt(str(x)+"_NLPP.txt")
    
    fig = plt.figure("CLBRES",figsize=(8.0,5.0))

    plt.loglog(CLB_data[:,0],CLB_data[:,1]*(CLB_data[:,0]**2)*((2.7*10**6)**2)/(2*np.pi),'b-')
    plt.ylim(10**-6,10**0)
    plt.xlabel(r"$L$",fontsize = 18)
    plt.ylabel(r"$\frac{L^2 C^{B B}_{l\ res}}{2\pi \mu K}$",fontsize = 18)
    plt.title(r"$C^{B B}_{l\ res}$"+' After '+str(x)+' iterations',fontsize = 20)
    plt.show()

    fig.savefig('C Iterative '+str(x)+' Iterations.jpg', dpi=fig.dpi)
    
    
    fig = plt.figure("NLPP",figsize=(8.0,5.0))

    plt.loglog(NLPP_data[:,0],(NLPP_data[:,0]**4)*NLPP_data[:,1]/(2*np.pi),'g-')
    plt.ylim(10**-9,2*10**-6)
    plt.xlabel(r"$L$",fontsize = 18)
    plt.ylabel(r"$\frac{L^4 N^{\phi\phi}_l}{2\pi \mu K}$",fontsize = 18)
    plt.title(r"$N^{\phi\phi}_l$"+'After '+str(x)+' Iterations',fontsize = 20)

    fig.savefig('N Iterative '+str(x)+' Iterations.jpg', dpi=fig.dpi)
    
    return print('done')


# In[13]:


Plot_Iteration(4)

