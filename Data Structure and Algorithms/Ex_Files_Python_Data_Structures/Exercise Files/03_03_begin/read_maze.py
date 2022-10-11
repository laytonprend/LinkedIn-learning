"""
Python Data Structures - A Game-Based Approach
Reading a maze from a text file
Robin Andrews - https://compucademy.net/
"""


def read_maze(file_name):
    """
    Reads a maze stored in a text file and returns a 2d list containing the maze representation.
    """
    f = open(file_name, "r")
    
    arr=[]
    for line in f:
      ##  try:
        line=line.replace('\n','')
       # arr.append(line)
     #   except exception as e:
      #      print('/')
        
       # print(line)
        LineArr=[]
        for square in line:
           # print(square)
            LineArr.append(square)
        arr.append(LineArr)
    print(arr)
    for i in arr:
        print(i)
        
    try:
        with open(file_name) as fh:
  
            maze = [[char for char in line.strip("\n")] for line in fh]
       
            num_cols_top_row = len(maze[0])
            Rectangle=True
            for row in maze:
                
                if len(row) != num_cols_top_row:
                    if Rectangle!=False:
                        Rectangle=False
                    print("The maze is not rectangular.")
                    raise SystemExit
                
            if Rectangle==True:
                print('The maze is rectangular')
                 ##   raise SystemExit
            return maze
        print('end')
    except OSError:
        print("There is a problem with the file you have selected.")
        raise SystemExit
        
if __name__=='__main__':
    print('START')
    #f = open("C://Users/layto/OneDrive/Documents/GitHub/LinkedIn-learning/Data Structure and Algorithms/Ex_Files_Python_Data_Structures/Exercise Files/03_03_begin/mazes/challenge_maze.txt", "r")
    
    read_maze("C://Users/layto/OneDrive/Documents/GitHub/LinkedIn-learning/Data Structure and Algorithms/Ex_Files_Python_Data_Structures/Exercise Files/03_03_begin/mazes/challenge_maze.txt")
            
    #C:\Users\layto\OneDrive\Documents\GitHub\LinkedIn-learning\Data Structure and Algorithms\Ex_Files_Python_Data_Structures\Exercise Files\03_03_begin\mazes
 #   with open("C:\\Users\layto\OneDrive\Documents\GitHub\LinkedIn-learning\Data Structure and Algorithms\Ex_Files_Python_Data_Structures\Exercise Files\03_03_begin\mazes\challenge_maze.txt",'r') as fd:
        
        #fd.method
  #      print('hallo')






