# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 18:41:36 2022

@author: layto
"""

import Stack

string='gninrael nideknil htiw tol a nrael'
reversed_string=''
s= Stack.Stack() 
for i in string:
    s.push(i)
while s.is_empty()==False:
    reversed_string+=s.pop()

    


print(reversed_string)

