
# coding: utf-8

# In[12]:


"""All imports"""
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import wigner_6j
from scipy.interpolate import interp1d
from math import sqrt, pi, exp, log, log10, factorial
import numpy as np
import matplotlib as mat
import time


# In[40]:


'''All Equations that do not rely on summations'''

data = np.loadtxt("Camb_Smooth.txt")

# C^EE from Camb Data
def C_EE(l): 
    return data[l-2,2]*(2*np.pi)/((l*(l+1))*((2.7*10**6)**2))

# C^ phi phi from Camb Data
def C_phi_phi(l): 
    return data[l-2,4]/((l**4)*((2.7*10**6)**2))

# F^s_lalblc Parts of the coubling coefficient eq. 7
def F(la,lb,lc,w):
    return (lb*(lb+1)+lc*(lc+1)-la*(la+1))*sqrt((2*lb+1)*(2*la+1)*(2*lc+1)/(16*pi))*w

# F^EB_lalblc Coupling coefficient CMB Eq. 6, w assumes s = 2. w(s=-2) = -w(s=2)
def f(la,lb,lc,w): 
    return (F(la,lb,lc,w)-F(la,lb,lc,-w))/(2)

def CLB_lensed_term(la,lb,lc,w): #The inner part of the summation of CMB Eq. 8 
    return C_phi_phi(lc)*C_EE(lb)*(f(la,lb,lc,w)**2)

def NBBEE(l): #Noise of BB and EE, which are equal eq 16
    arcmin = np.pi/(60*180)
    return (2*arcmin/(2.7*10**6))**2.*np.exp(((l*(5.*arcmin))**2.)/(8.*np.log(2.)))

def Nlpp_17_term(la,lb,lc,w,CLB): # Term of our first Nlpp sum eq 17
    return (f(la,lb,lc,w)**2)*(C_EE(lb)**2)/((CLB+NBBEE(la))*(C_EE(lb)+NBBEE(lb)))

def CLB_res_term(la,lb,lc,w,NLP):
    return (f(la,lb,lc,w)**2)*(C_EE(lb)*C_phi_phi(lc)+((C_EE(lb)**2)/(C_EE(lb)+NBBEE(lb)))*((C_phi_phi(lc)**2)/(C_phi_phi(lc)+NLP)))

def Nlpp_2_term(la,lb,lc,w,CLB):
    return(f(la,lb,lc,w)**2)*(1/(CLB+NBBEE(lb)))*((C_EE(lb)**2)/(C_EE(lb)+C_phi_phi(lb)))


# In[41]:


"""CLB_0  and NLPP_0 ... the rough estimation to get the ball rolling"""

def CLB_0(l1):
    Limit = 2500 # This is the highest l that I want to go to
    l3 = l1+2
    l2 = 2
    m1 = 2
    m2 = -2
    m3 = 0
    w = 0
    Wigner_Number = 0
    Wigner_Data = np.loadtxt("wigner_for_Clbb_smooth_"+str(int(l1))+".txt")
    CLB_lensed_sum = 0. # the value of the sum in Eq. 8. 
    
    """Outer loop starts at minimum l2 and goes up to l2 max
       Inner loop starts at maximum l3 and goes down
       because my wigner functions affect the final l,
       the order of l2 and l3 switches between loops 
       in a way that does not affect the value of w"""
    
    while (l2 <= Limit+l1): #L's order 213
        while (l3 >= abs(l1-l2)): #L's order 312
            
            #Adding a term to the sum
            if ((l1+l2+l3)%2 == 1 and l3 <= Limit+l1 and l3 <= 2500 and l2 <= 2500):
                w = Wigner_Data[Wigner_Number,1]
                CLB_lensed_sum += CLB_lensed_term(l1,l2,int(l3),w)
                Wigner_Number += 1
    
            l3 -= 1
        #Resets process with l2 being one greater    
        l2 +=1
        l3 = l1+l2    
    
    return (CLB_lensed_sum/(2*l1+1))





def NLPP_0(l3):
    
    CLB_f = CLB_function(0)
    Limit = 2500
    l2 = l3 + 2
    l1 = 2
    Nlpp_17_sum = 0 # the value of the sum in Eq. 34. 
    Wigner_Number = 0
    Wigner_Data = np.loadtxt("wigner_for_Clbb_L3_"+str(int(l3))+".txt")
  
    while (l1 <= Limit + l3): #213
        if l1 <= 2500:
            CLB = CLB_f(l1)
        while (l2 > abs(l3-l1) and l2 >= 2): #312
            if ((l1+l2+l3)%2 == 1 and l2 <= Limit+l3 and l2 <= 2500 and l1 <= 2500):
                w = Wigner_Data[Wigner_Number,1]
                Nlpp_17_sum += Nlpp_17_term(l1,l2,l3,w,CLB)
                Wigner_Number += 1
            
            l2 -= 1
            
        l1 +=1
        l2 = l3+l1    

    return (2*l3+1)/(Nlpp_17_sum)


# In[42]:


def CLB_function(X):
    CLB_data = np.loadtxt(str(X)+"_CLB.txt")
    return interp1d(CLB_data[:,0], CLB_data[:,1], kind='cubic')

def NLPP_function(X):   
    NLPP_data = np.loadtxt(str(X)+"_NLPP.txt")
    return interp1d(NLPP_data[:,0], NLPP_data[:,1], kind='cubic')


def NLPP_X(l3,X):
    
    if X == 0:
        return NLPP_0(l3)

    CLB_f = CLB_function(X)
    Limit = 2500
    l2 = l3 + 2
    l1 = 2
    Nlpp_2_sum = 0 # the value of the sum in Eq. 34. 
    Wigner_Number = 0
    Wigner_Data = np.loadtxt("wigner_for_Clbb_L3_"+str(int(l3))+".txt")
  
    while (l1 <= Limit + l3): #213
        if l1 <= 2500:
            CLB = CLB_f(l1)
        while (l2 > abs(l3-l1) and l2 >= 2): #312
            if ((l1+l2+l3)%2 == 1 and l2 <= Limit+l3 and l2 <= 2500 and l1 <= 2500):
                w = Wigner_Data[Wigner_Number,1]
                Nlpp_2_sum += Nlpp_2_term(l1,l2,l3,w,CLB)
                Wigner_Number += 1
            
            l2 -= 1
            
        l1 +=1
        l2 = l3+l1    

    return (2*l3+1)/(Nlpp_2_sum)


def CLB_X(l1,X):
    
    if X == 0:
        return CLB_0(l1)
    
    NLPP_f = NLPP_function(X-1)
    Limit = 2500 # This is the highest l that I want to go to
    l3 = l1+2
    l2 = 2
    m1 = 2
    m2 = -2
    m3 = 0
    w = 0
    Wigner_Number = 0.
    Wigner_Data = np.loadtxt("wigner_for_Clbb_smooth_"+str(int(l1))+".txt")
    CLB_2_sum = 0. # the value of the sum in Eq. 8. 
    
    """Outer loop starts at minimum l2 and goes up to l2 max
       Inner loop starts at maximum l3 and goes down
       because my wigner functions affect the final l,
       the order of l2 and l3 switches between loops 
       in a way that does not affect the value of w"""
    
    while (l2 <= Limit+l1): #L's order 213
        while (l3 >= abs(l1-l2)): #L's order 312
            if l3 >= 2 and l3 <= 2500:
                NLP = NLPP_f(l3)
            #Adding a term to the sum
            if ((l1+l2+l3)%2 == 1 and l3 <= Limit+l1 and l3 <= 2500 and l2 <= 2500):
                w = Wigner_Data[int(Wigner_Number),1]
                CLB_2_sum += CLB_res_term(l1,int(l2),int(l3),w,NLP)
                Wigner_Number += 1
    
            l3 -= 1
        #Resets process with l2 being one greater    
        l2 +=1
        l3 = l1+l2    
    
    return (CLB_2_sum/(2*l1+1))


# In[43]:


for x in range(5):
    
    Start = time.time()
    x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 13.0, 15.0, 17.0, 21.0, 25.0, 31.0, 38.0, 47.0, 58.0, 72.0, 89.0, 111.0, 139.0, 173.0, 217.0, 272.0, 342.0, 429.0, 539.0, 677.0, 851.0, 1070.0, 1346.0, 1694.0, 2131.0, 2500]
    CLB_y_axis = []

    for i in x_axis: 
        CLB_y_axis.append(CLB_X(int(i),x))
        
    CLB_values = []

    for i in range(len(x_axis)):
        CLB_values.append([x_axis[i],CLB_y_axis[i]])
    
    CLB_array = np.array(CLB_values)
    np.savetxt(str(x)+"_CLB.txt", CLB_array, delimiter=" ")
    
    Time = time.time() - Start
    Time /= 60
    print('CLB for ',x,' took ',Time,' minutes')
    
    
    
    Start = time.time()
    NLPP_y_axis = []

    for i in x_axis: 
        NLPP_y_axis.append(NLPP_X(int(i),x))
        
    NLPP_values = []

    for i in range(len(x_axis)):
        NLPP_values.append([x_axis[i],NLPP_y_axis[i]])
    
    NLPP_array = np.array(NLPP_values)
    np.savetxt(str(x)+"_NLPP.txt", NLPP_array, delimiter=" ")
    
    Time = time.time() - Start
    Time /= 60
    print('CLB for ',x,' took ',Time,' minutes')
    
    


# In[ ]:


[2, 5, 7, 9, 12, 16, 21, 28, 37, 49, 65, 87, 116, 155, 207, 276, 368, 491, 655, 873, 1164, 1552, 2070, 2500]

