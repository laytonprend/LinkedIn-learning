"""
Python Data Structures - A Game-Based Approach
Priority Queue Class based on heapq.
Robin Andrews - https://compucademy.net/
"""

import heapq


class PriorityQueue:
    def __init__(self):
        self.elements=[]
    def is_empty(self):
        return not self.elements
    def put(self,item,priority):
        heapq.heappush(self.elements,(priority,item))
    def get(self):
        return heapq.heappop(self.elements)[1]#1 is to print the word part
    def __str__(self):
        return(str(self.elements))
    
if __name__=='__main__':
    pq=PriorityQueue()

    print(pq)
    print(pq.is_empty())        
    pq.put('eat',2)
    pq.put('eat',2)
    pq.put('eat',2)#ctrl alt down to duplicate
    pq.put('sleep',3)#ctrl alt down to duplicate
    pq.put('code',1)#ctrl alt down to duplicate
    print(pq)#not ordered for print, highest priority printed
    print(pq.get())
    #inner working not our business, point of abstract data type
    #outputs lowest priority
    
    #only first element will be highest priority
    
        
        
        
