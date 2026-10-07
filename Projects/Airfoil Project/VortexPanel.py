"""
    This program uses the vortex-panel method outlined in Section 1.6 from Mechanics of Flight by Warren Phillips
    to compute the lift coefficient, leading-edge pitching-moment coefficient, and quarter-chord pitching-moment 
    coefficient at a specified angle of attack on an airfoil defined by a set of points that are read in from a 
    text file. 
"""

import json
import csv
import numpy as np
import math as m

def main():
    """ This is the main function to run the program. """
    # Import data from the JSON file
    data = JSON()

def JSON():
    """ This imports content from the JSON file. """
    with open('', 'r') as file:
        data = json.load(file)

    return data

def solutions():
    """ This function displays the results. """

if __name__ == "__main__":
    main()