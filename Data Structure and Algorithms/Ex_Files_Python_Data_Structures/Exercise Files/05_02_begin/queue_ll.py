"""
Python Data Structures - A Game-Based Approach
Queue class
Robin Andrews - https://compucademy.net/
"""

from collections import deque
#FIFO first in first out

class Queue:#lists aren't efficient for this purpose so we use a deque, optimises changes to start of list
    def __init__(self):
        self.items=deque()
    def is_empty(self):
        return not self.items#empty deque returns false, could use len==0
    def enqueue(self,item):
        self.items.append(item)#added to right hand side of deque
    def dequeue(self):
        return self.items.popleft()#remove far left
    def size(self):
        return len(self.items)
    def peek(self):
        return self.items[0]
    def __str__(self):#so is do print(q) it format right
        return str(self.items)
if __name__=='__main__':
    q=Queue()
    print(q)
    print(q.is_empty())
    q.enqueue('A')#duplicate ctrl alt down
    q.enqueue('B')#duplicate ctrl alt down
    q.enqueue('C')#duplicate ctrl alt down
    q.enqueue('D')#duplicate ctrl alt down
    q.enqueue('E')#duplicate ctrl alt down
    print(q)
    print(q.dequeue())#dequeue returns as popleft returns value removed
    print(q.size())
    print(q.peek())
    print(q)
    #challenge
    

    
