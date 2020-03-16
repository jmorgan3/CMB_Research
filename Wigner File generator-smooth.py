
# coding: utf-8

# In[81]:


"""This notebook will generate the Wigners needed for the selected interpolation points"""


# In[82]:


"""All imports"""
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import wigner_6j
from scipy.interpolate import interp1d
from math import sqrt, pi, exp, log, log10, factorial
import numpy as np
import matplotlib as mat
import time


# In[83]:


"""Iteration Functions"""
def Wigner_A(la,lb,lc,ma): # Math Eq. 6b, component of later functions
    return np.sqrt((-(lb-lc)**2)+la**2)*np.sqrt((((lb+lc+1)**2)-la**2))*np.sqrt((+la**2)-(ma**2))

def Wigner_B(la,lb,lc,ma,mb,mc): # Math Eq. 6c, component of later functions
    return -(2*la+1)*(lb*(lb+1)*ma-lc*(lc+1)*ma-la*(la+1)*(mc-mb))

def Wigner_Down(wa,wb,la,lb,lc,ma,mb,mc): # uses w(l+1) and w(l) to give w(l-1)
    return (la*Wigner_A(la+1,lb,lc,ma)*wb + Wigner_B(la,lb,lc,ma,mb,mc)*wa)/(-Wigner_A(la,lb,lc,ma)*(la+1))

def Wigner_Down_Limit(wa,la,lb,lc,ma,mb,mc): # uses w(lmax) to get w(lmax-1)
    return Wigner_B(la,lb,lc,ma,mb,mc)*wa/(-Wigner_A(la,lb,lc,ma)*(la+1))

def Wigner_Up(wa,wb,la,lb,lc,ma,mb,mc): # uses w(l-1) and w(l) to give w(l+1)
    return ((Wigner_B(la,lb,lc,ma,mb,mc)*wa) + (Wigner_A(la,lb,lc,ma)*(la+1)*wb))/(-la*Wigner_A(la+1,lb,lc,ma))

def Wigner_Up_Limit(wa,la,lb,lc,ma,mb,mc): # uses w(lmin) to get w(lmin+1)
    return (Wigner_B(la,lb,lc,ma,mb,mc)*wa)/(-la*Wigner_A(la+1,lb,lc,ma))

def Wigner_Max(la,lb,ma,mb,mc): # Calculates w of l = L+lp
    return ((-1)**(la-lb+mc))*sqrt((factorial(2*la)*factorial(2*lb)*factorial(mc+la+lb)*factorial(-mc+la+lb))/(factorial(2*la+2*lb+1)*factorial(la+ma)*factorial(la-ma)*factorial(lb+mb)*factorial(lb-mb)))

def Wigner_Max_Up(la,lb,ma,mb,mc): # factor (l3+l2,l2,l3+1)/(lp+L,L,lp)
    return -sqrt((2*la*(2*la-1)*(((la+lb)**2)-(mc**2)))/((2*la+2*lb+1)*(2*la+2*lb)*(la+ma)*(la-ma)))


# In[12]:


"""Take 2"""

"""Equation 8, iteratively calculated to dodge wigner's"""
def Wigner_Generator_L1():
    Start = time.time()
    x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 
              19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 
              58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 
              173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 
              481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 
              1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
    for l1 in x_axis:
        L1_Array=[]
        L2_Array=[]
        L3_Array=[]
        Wigner_Array = []
        Limit = 2500 # This is the highest l that I want to go to
        l3 = l1+2
        l2 = 2
        m1 = 2
        m2 = -2
        m3 = 0
        W = Wigner_Max(l2,l1,m2,m1,m3)   # this is the Wigner of l3(max), where each of the inner loops start   
        w = W # this variable holds the wigner coefficient
        w_current = W # Used to calculate the next w
        w_previous = 0 # Used to calculate the next w
        counter = 0

        """Outer loop starts at minimum l2 and goes up to l2 max
           Inner loop starts at maximum l3 and goes down
           because my wigner functions affect the final l,
           the order of l2 and l3 switches between loops 
           in a way that does not affect the value of w"""

        while (l2 <= Limit+l1): #L's order 213
            while (l3 >= abs(l1-l2)): #L's order 312
                #Adding a term to the sum
                if ((l1+l2+l3)%2 == 1 and l3 <= Limit+l1 and l3 <= 2500 and l2 <= 2500):
                    L1_Array.append(int(l1))
                    L2_Array.append(int(l2))
                    L3_Array.append(int(l3))
                    Wigner_Array.append(w)
                    counter += 1
                    
                    #if (l1+l2+l3) % 1020 == 1:
                        #print(float(w/wigner_3j(l1,l2,l3,2,-2,0)))
                #First time through the inner loop, no w_previous
                if (l3 == l1+l2):
                    w_previous = w
                    w = Wigner_Down_Limit(w,l3,l1,l2,m3,m1,m2)
                    w_current = w
                #all other times through inner loop
                if (l3 < l1+l2 and l3 > abs(l1-l2)): 
                    w = Wigner_Down(w_current,w_previous,l3,l1,l2,m3,m1,m2)
                    w_previous = w_current
                    w_current = w
                l3 -= 1
            #Resets process with l2 being one greater    
            l2 +=1
            l3 = l1+l2    
            W *= Wigner_Max_Up(l2,l1,m2,m1,m3) #Takes W to the new starting point
            w = W # puts w at the new starting point
        
        print('L1 of ', l1, 'is done')

        File = open('wigner_for_Clbb_smooth_'+str(int(l1))+'.txt', 'wb')
        Final_Array = []

        for i in range(0, counter, 1):
            Final_Array.append([L2_Array[i],Wigner_Array[i]])

        Final_Array_NP = np.array(Final_Array)
        np.savetxt("wigner_for_Clbb_smooth_"+str(int(l1))+".txt", Final_Array_NP, delimiter=" ")

        File.close

        L1_Array=[]
        L2_Array=[]
        L3_Array=[]
        Wigner_Array = []
    
    Time = time.time - Start
    return print('This cell took ', Time)

Wigner_Generator_L1()


# In[22]:


Start = time.time()
print(CLB_lensed(11.0))
Time = time.time()-Start
print('Cell took ', Time)


# In[182]:


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

l = 12
lb = 14
lc = 14
w = wigner_3j(la, lb, lc, 2, -2, 0)
w1 = wigner_3j(la, lb, lc, -2, 2, 0)

C_EELL = []
C_PPLL = []

for i in data[:,0]:
    C_EELL.append(C_EE(int(i)))
    C_PPLL.append(C_phi_phi(int(i)))

plt.loglog(data[:,0],(data[:,0]**2)*((2.7*10**6)**2)*C_EELL/((2*np.pi)))
plt.loglog(data[:,0],(data[:,0]**4)*C_PPLL/((2*np.pi)))


# In[169]:


"""Finds CLB from Wigner files"""

"""Equation 8, iteratively calculated to dodge wigner's"""
def CLB_lensed(l1):
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

l1=1070
Answer = CLB_lensed(l1)

print('answer is',Answer)


# In[172]:


print(l1*(l1+1)*Answer/(2*pi*(2.7*10**6)**2))


# In[236]:


"""Evaluates CLB_lensed at certain points for interpolation/plotting purposes"""
Start = time.time()
CLB_res_x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
CLB_res_y_axis = []

for i in CLB_res_x_axis:
    CLB_res_y_axis.append(CLB_lensed(i))
    print(i)
    
Time = time.time() - Start
print('This cell took ',Time,' seconds')


# In[237]:


"""Creates array of CLB lensed values and saves it as a text file"""
CLB_lensed_values = []

for i in range(len(CLB_lensed_x_axis)):
    CLB_lensed_values.append([CLB_lensed_x_axis[i],CLB_lensed_y_axis[i]])
    
CLB_lensed_array = np.array(CLB_lensed_values)
np.savetxt("CLB_lensed_smooth.txt", CLB_lensed_array, delimiter=" ")


# In[238]:


"""Create log plot of CLB_res in a PDF"""
CLB_lensed_data = np.loadtxt("CLB_lensed_smooth.txt")

fig = plt.figure("CLB_lensed")
plt.loglog(CLB_lensed_data[:,0],((CLB_lensed_data[:,0]**2)*(2.7*10**6)**2)/(2*pi)*CLB_lensed_data[:,1])
plt.title("CLB_lensed")
fig.savefig('CLB_lensed_smooth.pdf', dpi=fig.dpi)


# In[239]:


"""Creates an interpolated function using CLB lensed values"""

CLB_lensed_function = interp1d(CLB_lensed_data[:,0], CLB_lensed_data[:,1], kind='cubic')


# In[33]:


"""Generates Wigner Files for l3"""

def Wigner_Generator_L3():
    Start = time.time()
    x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
    for l3 in x_axis:    
        L1_Array=[]
        L2_Array=[]
        L3_Array=[]
        Wigner_Array = []
        Limit = 2500
        l2 = l3 + 2
        l1 = 2
        m1 = 2
        m2 = -2
        m3 = 0
        W = wigner_3j(l1,l3,l2,m1,m3,m2) #WMAX(l1,l3,m1,m3,m2)   # X, Y, and W are my "starting points" of each line   
        w = W # the wigner coefficient
        w_current = W
        w_previous = 0
        counter = 0

        while (l1 <= Limit+l3): #213
            while (l2 > abs(l3-l1) and l2 >= 2): #312
                if ((l1+l2+l3)%2 == 1 and l2 <= Limit+l3 and l2 <= 2500 and l1 <= 2500):
                    L1_Array.append(int(l1))
                    L2_Array.append(int(l2))
                    L3_Array.append(int(l3))
                    Wigner_Array.append(w)
                    counter += 1
                    
                if (l2 == l3+l1):
                    w_previous = w
                    w = Wigner_Down_Limit(w,l2,l3,l1,m2,m3,m1)
                    w_current = w

                if (l2 < l3+l1 and l2 > abs(l3-l1)):
                    w = Wigner_Down(w_current,w_previous,l2,l3,l1,m2,m3,m1)
                    w_previous = w_current
                    w_current = w
                l2 -= 1

            l1 +=1
            l2 = l3+l1    
            W *= Wigner_Max_Up(l1,l3,m1,m3,m2)
            w = W

        print('L3 of ', l3, 'is done')

        File = open('wigner_for_Clbb_L3_'+str(int(l3))+'.txt', 'wb')
        Final_Array = []

        for i in range(0, counter, 1):
            Final_Array.append([L2_Array[i],Wigner_Array[i]])

        Final_Array_NP = np.array(Final_Array)
        np.savetxt("wigner_for_Clbb_L3_"+str(int(l3))+".txt", Final_Array_NP, delimiter=" ")

        File.close

        L1_Array=[]
        L2_Array=[]
        L3_Array=[]
        Wigner_Array = []
        
        """if l3 == 2500:
            break
        l3 *= 10**.4
        l3  //= 1

        if l3 > 2500:
            break"""
    
    Time = time.time() - Start
    print('This cell took ',Time,' seconds')
    return print('¯\_(ツ)_/¯')


Wigner_Generator_L3()


# In[240]:


"""New eauations needed for Noise estimation"""
def NBBEE(l): #Noise of BB and EE, which are equal eq 16
    arcmin = np.pi/(60*180)
    return (2*arcmin/(2.7*10**6))**2.*np.exp(((l*(5.*arcmin))**2.)/(8.*np.log(2.)))

def Nlpp_17_term(la,lb,lc,w,CLB): # Term of our first Nlpp sum eq 17
    return (f(la,lb,lc,w)**2)*(C_EE(lb)**2)/((CLB+NBBEE(la))*(C_EE(lb)+NBBEE(lb)))


# In[241]:


def Nlpp_17(l3):
    Limit = 2500
    l2 = l3 + 2
    l1 = 2
    Nlpp_17_sum = 0 # the value of the sum in Eq. 34. 
    Wigner_Number = 0
    Wigner_Data = np.loadtxt("wigner_for_Clbb_L3_"+str(int(l3))+".txt")
  
    while (l1 <= Limit + l3): #213
        if l1 <= 2500:
            CLB = CLB_lensed_function(l1)
        while (l2 > abs(l3-l1) and l2 >= 2): #312
            if ((l1+l2+l3)%2 == 1 and l2 <= Limit+l3 and l2 <= 2500 and l1 <= 2500):
                w = Wigner_Data[Wigner_Number,1]
                Nlpp_17_sum += Nlpp_17_term(l1,l2,l3,w,CLB)
                Wigner_Number += 1
            
            l2 -= 1
            
        l1 +=1
        l2 = l3+l1    

    return (2*l3+1)/(Nlpp_17_sum)

print(Nlpp_17(1200))


# In[242]:


"""Evaluates Nlpp_17 at certain points for interpolation/plotting purposes"""
Start = time.time()
Nlpp_17_x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
Nlpp_17_y_axis = []

for i in Nlpp_17_x_axis: 
    Nlpp_17_y_axis.append(Nlpp_17(int(i)))
    print(i)
Time = time.time() - Start
print('This cell took ',Time,' seconds')


# In[243]:


"""Creates array of Nlpp_17 values and saves it as a text file"""
Nlpp_17_values = []

for i in range(len(Nlpp_17_x_axis)):
    Nlpp_17_values.append([Nlpp_17_x_axis[i],Nlpp_17_y_axis[i]])
    
Nlpp_17_array = np.array(Nlpp_17_values)
np.savetxt("Nlpp_17_smooth.txt", Nlpp_17_array, delimiter=" ")


# In[244]:


"""Create log plot of Nlpp_17 in a PDF"""
Nlpp_17_data = np.loadtxt("Nlpp_17_smooth.txt")

fig = plt.figure("Nlpp_17")
plt.loglog(Nlpp_17_data[:,0],(Nlpp_17_data[:,0]**4)*Nlpp_17_data[:,1]/(2*np.pi))
plt.title("Nlpp_17")
fig.savefig('Nlpp_17_smooth.pdf', dpi=fig.dpi)


# In[245]:


'''new definitions for equation 18 CLB_res'''

"""Creates an interpolated function using NLpp_17 values"""

Nlpp_17_function = interp1d(Nlpp_17_data[:,0], Nlpp_17_data[:,1], kind='cubic')

def CLB_res_term(la,lb,lc,w,NLP):
    return (f(la,lb,lc,w)**2)*(C_EE(lb)*C_phi_phi(lc)+((C_EE(lb)**2)/(C_EE(lb)+NBBEE(lb)))*((C_phi_phi(lc)**2)/(C_phi_phi(lc)+NLP)))


# In[246]:


Nlpp_17_function = interp1d(Nlpp_17_data[:,0], Nlpp_17_data[:,1], kind='cubic')


def CLB_res(l1):
    Limit = 2500 # This is the highest l that I want to go to
    l3 = l1+2
    l2 = 2
    m1 = 2
    m2 = -2
    m3 = 0
    w = 0
    Wigner_Number = 0.
    Wigner_Data = np.loadtxt("wigner_for_Clbb_smooth_"+str(int(l1))+".txt")
    CLB_res_sum = 0. # the value of the sum in Eq. 8. 
    
    """Outer loop starts at minimum l2 and goes up to l2 max
       Inner loop starts at maximum l3 and goes down
       because my wigner functions affect the final l,
       the order of l2 and l3 switches between loops 
       in a way that does not affect the value of w"""
    
    while (l2 <= Limit+l1): #L's order 213
        while (l3 >= abs(l1-l2)): #L's order 312
            if l3 >= 2 and l3 <= 2500:
                NLP = Nlpp_17_function(l3)
            #Adding a term to the sum
            if ((l1+l2+l3)%2 == 1 and l3 <= Limit+l1 and l3 <= 2500 and l2 <= 2500):
                w = Wigner_Data[int(Wigner_Number),1]
                CLB_res_sum += CLB_res_term(l1,int(l2),int(l3),w,NLP)
                Wigner_Number += 1
    
            l3 -= 1
        #Resets process with l2 being one greater    
        l2 +=1
        l3 = l1+l2    
    
    return (CLB_res_sum/(2*l1+1))

l1=11
Answer = CLB_res(l1)

print('answer is',Answer)


# In[247]:


"""Evaluates CLB_res at certain points for interpolation/plotting purposes"""
Start = time.time()
CLB_res_x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
CLB_res_y_axis = []

for i in CLB_res_x_axis: 
    CLB_res_y_axis.append(CLB_res(int(i)))
    print(i)
Time = time.time() - Start
print('This cell took ',Time,' seconds')


# In[248]:


"""Creates array of CLB_res values and saves it as a text file"""
CLB_res_values = []

for i in range(len(CLB_res_x_axis)):
    CLB_res_values.append([CLB_res_x_axis[i],CLB_res_y_axis[i]])
    
CLB_res_array = np.array(CLB_res_values)
np.savetxt("CLB_res_smooth.txt", CLB_res_array, delimiter=" ")


# In[263]:


"""Create log plot of CLB_res in a PDF"""
CLB_res_data = np.loadtxt("CLB_res_smooth.txt")

fig = plt.figure("CLB_res")
plt.loglog(CLB_res_data[:,0],CLB_res_data[:,1]*(CLB_res_data[:,0]**2)*((2.7*10**6)**2)/(2*pi))
plt.title("CLB_res")
fig.savefig('CLB_res_smooth.pdf', dpi=fig.dpi)


# In[251]:


def Nlpp_2_term(la,lb,lc,w,CLB):
    return(f(la,lb,lc,w)**2)*(1/(CLB+NBBEE(lb)))*((C_EE(lb)**2)/(C_EE(lb)+C_phi_phi(lb)))


# In[252]:


"""Uses Results from first CLB res to calculate refined Nlpp"""
CLB_res_data = np.loadtxt("CLB_res_smooth.txt")
CLB_res_function = interp1d(CLB_res_data[:,0], CLB_res_data[:,1], kind='cubic')


def Nlpp_2(l3):
    Limit = 2500
    l2 = l3 + 2
    l1 = 2
    Nlpp_2_sum = 0 # the value of the sum in Eq. 34. 
    Wigner_Number = 0
    Wigner_Data = np.loadtxt("wigner_for_Clbb_L3_"+str(int(l3))+".txt")
  
    while (l1 <= Limit + l3): #213
        if l1 <= 2500:
            CLB = CLB_res_function(l1)
        while (l2 > abs(l3-l1) and l2 >= 2): #312
            if ((l1+l2+l3)%2 == 1 and l2 <= Limit+l3 and l2 <= 2500 and l1 <= 2500):
                w = Wigner_Data[Wigner_Number,1]
                Nlpp_2_sum += Nlpp_2_term(l1,l2,l3,w,CLB)
                Wigner_Number += 1
            
            l2 -= 1
            
        l1 +=1
        l2 = l3+l1    

    return (2*l3+1)/(Nlpp_2_sum)

print(Nlpp_2(2))



# In[270]:


"""Evaluates Nlpp_2 at certain points for interpolation/plotting purposes"""
Start = time.time()
Nlpp_2_x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
Nlpp_2_y_axis = []

for i in Nlpp_2_x_axis: 
    Nlpp_2_y_axis.append(Nlpp_2(int(i)))
    print(i)
Time = time.time() - Start
print('This cell took ',Time,' seconds')


# In[271]:


"""Creates array of Nlpp_2 values and saves it as a text file"""
Nlpp_2_values = []

for i in range(len(Nlpp_2_x_axis)):
    Nlpp_2_values.append([Nlpp_2_x_axis[i],Nlpp_2_y_axis[i]])
    
Nlpp_2_array = np.array(Nlpp_2_values)
np.savetxt("Nlpp_2_smooth.txt", Nlpp_2_array, delimiter=" ")


# In[272]:


"""Create log plot of Nlpp_2 in a PDF"""
Nlpp_2_data = np.loadtxt("Nlpp_2_smooth.txt")

fig = plt.figure("Nlpp_2")
plt.loglog(Nlpp_2_data[:,0],(Nlpp_2_data[:,0]**4)*Nlpp_2_data[:,1]/(2*np.pi))
plt.title("Nlpp_2")
fig.savefig('Nlpp_2_smooth.pdf', dpi=fig.dpi)


# In[273]:


Nlpp_2_data = np.loadtxt("Nlpp_2_smooth.txt")
Nlpp_2_function = interp1d(Nlpp_2_data[:,0], Nlpp_2_data[:,1], kind='cubic')


# In[274]:


Nlpp_2_function = interp1d(Nlpp_2_data[:,0], Nlpp_2_data[:,1], kind='cubic')

def CLB_2(l1):
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
                NLP = Nlpp_2_function(l3)
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

l1=11
Answer = CLB_2(l1)

print('answer is',Answer)


# In[275]:


"""Evaluates CLB_2 at certain points for interpolation/plotting purposes"""
Start = time.time()
CLB_2_x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
CLB_2_y_axis = []

for i in CLB_2_x_axis: 
    CLB_2_y_axis.append(CLB_2(int(i)))
    print(i)
Time = time.time() - Start
print('This cell took ',Time,' seconds')


# In[276]:


"""Creates array of CLB_2 values and saves it as a text file"""
CLB_2_values = []

for i in range(len(CLB_2_x_axis)):
    CLB_2_values.append([CLB_2_x_axis[i],CLB_2_y_axis[i]])
    
CLB_2_array = np.array(CLB_2_values)
np.savetxt("CLB_2_smooth.txt", CLB_2_array, delimiter=" ")


# In[277]:


"""Create log plot of Nlpp_2 in a PDF"""
CLB_2_data = np.loadtxt("CLB_2_smooth.txt")

fig = plt.figure("CLB_2")
plt.loglog(CLB_2_data[:,0],CLB_2_data[:,1]*(CLB_2_data[:,0]**2)*((2.7*10**6)**2)/(2*pi),'b-')
plt.title("CLB_2")
fig.savefig('CLB_2_smooth.pdf', dpi=fig.dpi)


# In[326]:


fig = plt.figure("NLPP",figsize=(8.0,5.0))

plt.loglog(Nlpp_3_data[:,0],(Nlpp_3_data[:,0]**4)*Nlpp_3_data[:,1]/(2*np.pi),'g-')
plt.ylim(10**-9,2*10**-6)
plt.xlabel(r"$L$",fontsize = 18)
plt.ylabel(r"$\frac{L^4 N^{\phi\phi}_l}{2\pi \mu K}$",fontsize = 18)
plt.title(r"$N^{\phi\phi}_l$"+'After 3 Iterations',fontsize = 20)

fig.savefig('N 3 Iterations.jpg', dpi=fig.dpi)


# In[325]:


fig = plt.figure("CLBRES",figsize=(8.0,5.0))

plt.loglog(CLB_3_data[:,0],CLB_3_data[:,1]*(CLB_3_data[:,0]**2)*((2.7*10**6)**2)/(2*pi),'b-')
plt.ylim(10**-6,10**0)
plt.xlabel(r"$L$",fontsize = 18)
plt.ylabel(r"$\frac{L^2 C^{B B}_{l\ res}}{2\pi \mu K}$",fontsize = 18)
plt.title(r"$C^{B B}_{l\ res}$"+' After 3 iterations',fontsize = 20)

fig.savefig('C 3 Iterations.jpg', dpi=fig.dpi)


# In[323]:


'''Improvement factor'''

Alpha = []

CLB_3_data = np.loadtxt("CLB_3_smooth.txt")
Nlpp_3_data = np.loadtxt("Nlpp_3_smooth.txt")

for i in range(len(Nlpp_17_data)):
    a = (Nlpp_3_data[i,1] + CLB_3_data[i,1])/(Nlpp_2_data[i,1] + CLB_2_data[i,1])
    Alpha.append(a)

fig = plt.figure("Alpha",figsize=(8.0,5.0))    

plt.plot(CLB_lensed_data[:,0],Alpha[:])
plt.title('Improvement Factor for Third Iteration')
plt.xlabel('l')
plt.ylabel(r"$\frac{N^{\phi \phi}_{3} + C^{B B}_{3}}{N^{\phi \phi}_{2} + C^{B B}_{2}}$",fontsize = 18)
fig.savefig('Improvement_Factor.jpg', dpi=fig.dpi)


# In[302]:


"""Uses Results from CLB 2 to calculate Nlpp 3"""
CLB_2_data = np.loadtxt("CLB_2_smooth.txt")
CLB_2_function = interp1d(CLB_2_data[:,0], CLB_2_data[:,1], kind='cubic')


def Nlpp_3(l3):
    Limit = 2500
    l2 = l3 + 2
    l1 = 2
    Nlpp_3_sum = 0 # the value of the sum in Eq. 34. 
    Wigner_Number = 0
    Wigner_Data = np.loadtxt("wigner_for_Clbb_L3_"+str(int(l3))+".txt")
  
    while (l1 <= Limit + l3): #213
        if l1 <= 2500:
            CLB = CLB_2_function(l1)
        while (l2 > abs(l3-l1) and l2 >= 2): #312
            if ((l1+l2+l3)%2 == 1 and l2 <= Limit+l3 and l2 <= 2500 and l1 <= 2500):
                w = Wigner_Data[Wigner_Number,1]
                Nlpp_3_sum += Nlpp_2_term(l1,l2,l3,w,CLB)
                Wigner_Number += 1
            
            l2 -= 1
            
        l1 +=1
        l2 = l3+l1    

    return (2*l3+1)/(Nlpp_3_sum)


# In[305]:


"""Evaluates Nlpp_3 at certain points for interpolation/plotting purposes"""
Start = time.time()
Nlpp_3_x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
Nlpp_3_y_axis = []

for i in Nlpp_3_x_axis: 
    Nlpp_3_y_axis.append(Nlpp_3(int(i)))
    print(i)
Time = time.time() - Start
print('This cell took ',Time,' seconds')


# In[306]:


"""Creates array of Nlpp_3 values and saves it as a text file"""
Nlpp_3_values = []

for i in range(len(Nlpp_3_x_axis)):
    Nlpp_3_values.append([Nlpp_3_x_axis[i],Nlpp_3_y_axis[i]])
    
Nlpp_3_array = np.array(Nlpp_3_values)
np.savetxt("Nlpp_3_smooth.txt", Nlpp_3_array, delimiter=" ")


# In[310]:


'''Function for calculating CLB 3'''

Nlpp_3_data = np.loadtxt("Nlpp_3_smooth.txt")
Nlpp_3_function = interp1d(Nlpp_3_data[:,0], Nlpp_3_data[:,1], kind='cubic')


def CLB_3(l1):
    Limit = 2500 # This is the highest l that I want to go to
    l3 = l1+2
    l2 = 2
    m1 = 2
    m2 = -2
    m3 = 0
    w = 0
    Wigner_Number = 0.
    Wigner_Data = np.loadtxt("wigner_for_Clbb_smooth_"+str(int(l1))+".txt")
    CLB_3_sum = 0. # the value of the sum in Eq. 8. 
    
    """Outer loop starts at minimum l2 and goes up to l2 max
       Inner loop starts at maximum l3 and goes down
       because my wigner functions affect the final l,
       the order of l2 and l3 switches between loops 
       in a way that does not affect the value of w"""
    
    while (l2 <= Limit+l1): #L's order 213
        while (l3 >= abs(l1-l2)): #L's order 312
            if l3 >= 2 and l3 <= 2500:
                NLP = Nlpp_3_function(l3)
            #Adding a term to the sum
            if ((l1+l2+l3)%2 == 1 and l3 <= Limit+l1 and l3 <= 2500 and l2 <= 2500):
                w = Wigner_Data[int(Wigner_Number),1]
                CLB_3_sum += CLB_res_term(l1,int(l2),int(l3),w,NLP)
                Wigner_Number += 1
    
            l3 -= 1
        #Resets process with l2 being one greater    
        l2 +=1
        l3 = l1+l2    
    
    return (CLB_3_sum/(2*l1+1))

l1=11
Answer = CLB_3(l1)

print('answer is',Answer)


# In[311]:


"""Evaluates CLB_3 at certain points for interpolation/plotting purposes"""
Start = time.time()
CLB_3_x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
CLB_3_y_axis = []

for i in CLB_3_x_axis: 
    CLB_3_y_axis.append(CLB_3(int(i)))
    print(i)
Time = time.time() - Start
print('This cell took ',Time,' seconds')


# In[312]:


"""Creates array of CLB_3 values and saves it as a text file"""
CLB_3_values = []

for i in range(len(CLB_3_x_axis)):
    CLB_3_values.append([CLB_3_x_axis[i],CLB_3_y_axis[i]])
    
CLB_3_array = np.array(CLB_3_values)
np.savetxt("CLB_3_smooth.txt", CLB_3_array, delimiter=" ")


# In[313]:


"""Uses Results from CLB 3 to calculate Nlpp 4"""
CLB_3_data = np.loadtxt("CLB_3_smooth.txt")
CLB_3_function = interp1d(CLB_3_data[:,0], CLB_3_data[:,1], kind='cubic')


def Nlpp_4(l3):
    Limit = 2500
    l2 = l3 + 2
    l1 = 2
    Nlpp_4_sum = 0 # the value of the sum in Eq. 34. 
    Wigner_Number = 0
    Wigner_Data = np.loadtxt("wigner_for_Clbb_L3_"+str(int(l3))+".txt")
  
    while (l1 <= Limit + l3): #213
        if l1 <= 2500:
            CLB = CLB_3_function(l1)
        while (l2 > abs(l3-l1) and l2 >= 2): #312
            if ((l1+l2+l3)%2 == 1 and l2 <= Limit+l3 and l2 <= 2500 and l1 <= 2500):
                w = Wigner_Data[Wigner_Number,1]
                Nlpp_4_sum += Nlpp_2_term(l1,l2,l3,w,CLB)
                Wigner_Number += 1
            
            l2 -= 1
            
        l1 +=1
        l2 = l3+l1    

    return (2*l3+1)/(Nlpp_4_sum)


# In[314]:


"""Evaluates Nlpp_4 at certain points for interpolation/plotting purposes"""
Start = time.time()
Nlpp_4_x_axis = [2.0, 4.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 25.0, 28.0, 31.0, 34.0, 38.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0, 80.0, 89.0, 99.0, 111.0, 124.0, 139.0, 155.0, 173.0, 194.0, 217.0, 243.0, 272.0, 305.0, 342.0, 383.0, 429.0, 481.0, 539.0, 604.0, 677.0, 759.0, 851.0, 954.0, 1070.0, 1200.0, 1346.0, 1510.0, 1694.0, 1900.0, 2131.0, 2391.0, 2500]
Nlpp_4_y_axis = []

for i in Nlpp_4_x_axis: 
    Nlpp_4_y_axis.append(Nlpp_4(int(i)))
    print(i)
Time = time.time() - Start
print('This cell took ',Time,' seconds')

