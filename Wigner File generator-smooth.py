
# coding: utf-8

# In[81]:


"""This notebook will generate the Wigners needed for the selected interpolation points"""


# In[82]:


"""All imports"""
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
def Wigner_Generator_L1(EO,S):
    x_axis = [2, 3, 4, 5, 6, 8, 14, 19, 26, 36, 50, 70, 98, 138, 194, 274, 387, 546, 771, 1089, 1538, 2172, 3068, 4333, 5000]
    for l1 in x_axis:
        L1_Array=[]
        L2_Array=[]
        L3_Array=[]
        Wigner_Array = []
        Limit = 5000 # This is the highest l that I want to go to
        l3 = l1+2
        l2 = 2
        m1 = S
        m2 = -S
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
                if ((l1+l2+l3)%2 == int(EO) and l3 <= Limit+l1 and l3 <= 5000 and l2 <= 5000):
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

        File = open('L1/'+str(EO)+'/'+str(S)+'/'+str(Limit)+'_Wigners_for_L1_'+str(int(l1))+'.txt', 'wb')
        Final_Array = []

        for i in range(0, counter, 1):
            Final_Array.append([L2_Array[i],Wigner_Array[i]])

        Final_Array_NP = np.array(Final_Array)
        np.savetxt('L1/'+str(EO)+'/'+str(S)+'/'+str(Limit)+'_Wigners_for_L1_'+str(int(l1))+'.txt', Final_Array_NP, delimiter=" ")

        File.close

        L1_Array=[]
        L2_Array=[]
        L3_Array=[]
        Wigner_Array = []
    
    return print('Done')


"""Generates Wigner Files for l3"""

def Wigner_Generator_L3(EO,S):
    x_axis = [2, 3, 4, 5, 6, 8, 14, 19, 26, 36, 50, 70, 98, 138, 194, 274, 387, 546, 771, 1089, 1538, 2172, 3068, 4333, 5000]
    print('started')
    for l3 in x_axis:    
        L1_Array=[]
        L2_Array=[]
        L3_Array=[]
        Wigner_Array = []
        Limit = 5000
        l2 = l3 + 2
        l1 = 2
        m1 = S
        m2 = -S
        m3 = 0
        W = wigner_3j(l1,l3,l2,m1,m3,m2) #WMAX(l1,l3,m1,m3,m2)   # X, Y, and W are my "starting points" of each line   
        w = W # the wigner coefficient
        w_current = W
        w_previous = 0
        counter = 0
        Start = time.time()
        while (l1 <= Limit+l3): #213
            while (l2 > abs(l3-l1) and l2 >= 2): #312
                if ((l1+l2+l3)%2 == int(EO) and l2 <= Limit+l3 and l2 <= 5000 and l1 <= 5000):
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

        File = open('L3/'+str(EO)+'/'+str(S)+'/'+str(Limit)+'_Wigners_for_L3_'+str(int(l3))+'.txt', 'wb')
        Final_Array = []

        for i in range(0, counter, 1):
            Final_Array.append([L2_Array[i],Wigner_Array[i]])

        Final_Array_NP = np.array(Final_Array)
        np.savetxt('L3/'+str(EO)+'/'+str(S)+'/'+str(Limit)+'_Wigners_for_L3_'+str(int(l3))+'.txt', Final_Array_NP, delimiter=" ")

        File.close

        L1_Array=[]
        L2_Array=[]
        L3_Array=[]
        Wigner_Array = []
        
    
        Time = time.time() - Start
        print('This cell took ',Time,' seconds')
    return print('¯\_(ツ)_/¯')

Wigner_Generator_L1(0,2)
print('0,2 L1')
Wigner_Generator_L1(0,0)
print('0,0 L1')
Wigner_Generator_L1(1,2)
print('1,2 L1')'''

Wigner_Generator_L3(0,2)
print('0,2 L3')
Wigner_Generator_L3(0,0)
print('0,0 L3')
Wigner_Generator_L3(1,2)
print('1,2 L3')'''


