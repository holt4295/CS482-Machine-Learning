# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:47:28 2023

@author: gholt
"""

import numpy as np
a = np.array([2,3,4])
print("\nArray A:\n",a)
print("\nArray A Data Type:\n", a.dtype)
b = np.array([1.2, 3.5, 5.1])
print("\nArray B Data Type:\n", b.dtype)

# Exercise 1
e1 = np.array([1,2,3,4])
print("\nArray e1:\n",e1)
print("\nArray e1 Data Type:\n", e1.dtype) 


c = np.zeros((2,3))
d = np.ones((3,4))
e = np.empty((9))
print("\nArray C:\n",c)
print("\nArray C Data Type:\n", c.dtype)
print("\nArray D:\n",d)
print("\nArray D Data Type:\n", d.dtype)
print("\nArray E:\n",e)
print("\nArray E Data Type:\n", e.dtype)

# Exercise 2
e2 = np.zeros((2,7))
print("\nArray e2:\n",e2)
print("\nArray e2 Data Type:\n", e2.dtype)


f = np.arange( 10, 30, 5 )
g = np.linspace( 0, 2, 9 )
print("\nArray f:\n",f)
print("\nArray f Data Type:\n", f.dtype)
print("\nArray g:\n",g)
print("\nArray gData Type:\n", g.dtype)

# Exercise 3
e3 = np.arange(1, 23, 2.5)
print("\nArray e3:\n",e3)
print("\nArray e3 Data Type:\n", e3.dtype)


h = np.array([[1,2],
 [3,4],
 [5,6]])
print("\nArray H:\n",h)
print("\nValue at (1,1):\n",h[1,1])


i = np.array([[1,2,3,4],
 [5,6,7,8],
 [9,10,11,12]])
print("\nArray I:\n",i)
print("\nArray I - ravel:\n",i.ravel())
print("\nArray I - reshaped 2X6:\n",i.reshape(2,6))
print("\nArray I - transposed:\n",i.T)
print("\nArray I Shape:\n",i.shape)
print("\nArray I.T Shape:\n",i.T.shape)


# Exercise 4
e4 = np.arange(0, 40)
print("\nArray e4:\n",e4)
e4 = e4.reshape(5,8)
print("\nArray e4 - reshaped 5X8:\n",e4)
print("\nValue at (5,8):\n",e4[4,7])
