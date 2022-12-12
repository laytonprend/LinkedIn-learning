# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:21:40 2022

@author: layto
"""


class Stack:
    def __init__(self):#constructor when class is made
        self.items=[]
        
    def is_empty(self):
        return len(self.items)>0#not self.items#empty list evaluates as false in python
    def push(self,item):
        self.items.append(item)#adds an item
    def pop(self):
        return(self.items.pop()) #removes the first item on stack which is the last added       
    def peek(self):
        return self.items[-1]#see the first item on stack but leave it in place
    def size(self):
        return len(self.items)#length of Stack
    def __str__(self):
        return str(self.items)#print entire stack
string='hello'
half1=string[:int(len(string)/2)]
half2=string[int(len(string)/2)+1:]
print(half1)
print(half2)
reversehalf2=''
for i in half2:
    s=Stack()
    s.push(i)
    print('i',i)
while s.is_empty()==False:
    val=s.pop()
    reversehalf2=reversehalf2+val
    print('val',val)
    
print(half1)
print(half2)
print(reversehalf2)
#for x in string:
    
    