# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:45:51 2022

@author: layto
"""

class Stack:
    def __init__(self):#constructor when class is made
        self.items=[]
        
    def is_empty(self):
        return not self.items#empty list evaluates as false in python
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
    
if __name__ =='__main__':
    s=Stack()#calls constructor
    print(s)
    print(s.is_empty())
    s.push(3)
    print(s)
    s.push(7)
    s.push(8)
    print(s)
    print(s.pop())#first in last out, removes most recently added number first
    print(s)
    print(s.peek())
    print(s.size())
    print('print entire list',s.__str__())
    
    #print('new oop',
    ##oop , if there is values in there use {your_dog.name}
    
        