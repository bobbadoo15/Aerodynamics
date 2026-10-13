"""
    This program uses the vortex-panel method outlined in Section 1.6 from Mechanics of Flight by Warren Phillips
    to compute the lift coefficient, leading-edge pitching-moment coefficient, and quarter-chord pitching-moment 
    coefficient at a specified angle of attack on an airfoil defined by a set of points that are read in from a 
    text file. 

    Notes:
        4) Find the panel lengths; Eq. 1.6.19
        5) Find and store the location of each control point and store; Eq. 1.6.18
        6) 
"""

import json
import csv
import numpy as np
import math as m
from pathlib import Path

def main():
    """ This is the main function to run the program. """
    # Import data from the JSON file
    a, g, v = load_json_data()
    # Import data from the geometry
    data = load_geometry_data(g)
    # Find the relative nodes and panels for the given airfoil
    N, Nx, Ny, Np = nodes_and_panels(data)
    # Preallocate variables
    # l = preallocate(Nx,Ny)
    # Find the panel lengths
    l = panel_length(Nx,Ny)

    
    # Debugging
    # print(Nx)
    # print(l)

def load_geometry_data(g):
    """ This imports data from the geometry .txt file. """
    # Define the input path
    script_dir = Path(__file__).parent
    input_path = script_dir/g
    # Read in txt file of the geometry
    with open(input_path, 'r') as file:
        data = np.loadtxt(file)
    return data

def load_json_data():
    """ This imports all content from the .txt file. """
    # Define the input path
    script_dir = Path(__file__).parent
    input_path = script_dir/"input.json"
    # Open and load JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)
    # Pull JSON content out
    a = data["alpha[deg]"]
    v = data["freestream_velocity"]
    g = data["geometry"]
    return a, g, v

def nodes_and_panels(data):
    """ This finds the relative node points and number of panels. """
    # Store Node points
    Nx = data[:,0]
    Ny = data[:,1]
    # Find the total number of nodes
    N = len(Nx)
    # Find the total number of panels
    Np = N - 1
    return N, Nx, Ny, Np

def panel_length(Nx,Ny):
    """ This is where the length of the panels are calculated. """
    l = []
    for i in range(len(Nx)-1):
        length = np.sqrt((Nx[i+1]-Nx[i])**2 + (Ny[i+1]-Ny[i])**2)
        l.append(length)
    return l

def preallocate(Nx,Ny):
    """ This creates pre-allocated variables. """
    # l = np.zeros((len(Nx),len(Ny)))
    return l

if __name__ == "__main__":
    main()