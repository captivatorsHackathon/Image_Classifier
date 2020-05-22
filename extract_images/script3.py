# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:16:58 2020

@author: ABC
"""


# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
  
    for count, filename in enumerate(os.listdir("images")): 
        dst ="hawamahal." + str(count+1) + ".jpg"
        src ='images/'+ filename 
        dst ='images/'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 
