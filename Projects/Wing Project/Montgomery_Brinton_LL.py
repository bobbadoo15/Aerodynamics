'''
This program genearates a theoretical lifting line by using '...the numerical Fourier series 
solution to Prandtl's lifting-line theory to predict the aerodynamic characteristics of a 
finite wing with a given planform, washout, aileron deflection, and rigid-body roll.' The user 
can create a JSON file for a desired airfoil shape to use as the interested input file. 

Important equations used to solve for the desired LLT:
    Aspect Ratio: 
        R_A = b^2/S = b/ctilde
    Elliptic Chord Distribution: 
        c(z)/b = (4/piR_A)sqrt(1-(2z/b)^2)
'''

# ==========================================================================================
#                                        IMPORTS
# ==========================================================================================

import json
import matplotlib.pyplot as plt
import math as m
import numpy as np
import csv
from pathlib import Path
import sys

# ==========================================================================================
#                                          MAIN
# ==========================================================================================

def main(pi):
    """ This contains the structure of the program. """
    # Import JSON content from 'input.json'
    data = load_input_data()
    # Create the number of nodes along the semispan
    total_nodes = node(data,pi)
    # Calculate for the theta values
    theta, zpb = cov(total_nodes,pi)
    # Define the shape of the wing
    geometry(data,zpb,pi)
    # Find the results
    results(data)
    # Plot results
    # PlotLL(zpb)

# ==========================================================================================
#                                       NODAL COUNT
# ==========================================================================================

def node(data,pi):
    """ This function determines the node count the user desires. """
    # Import the total number of desired nodes
    nodes = data["wing"]["nodes_per_semispan"]
    # Find the total nodes for the entire span
    total_nodes = (2*nodes) - 1

    return total_nodes

# ==========================================================================================
#                                  CHANGE OF VARIABLES
# ==========================================================================================

def cov(total_nodes,pi):
    """ This function finds the change-of-variables (theta) for cosine clustering. """
    # Generate theta values
    theta = np.linspace(0,pi,total_nodes)
    # Find the related z/b values
    zpb = -0.5 * np.cos(theta)

    return theta, zpb

# ==========================================================================================
#                            IMPORTING "input.json" INFORMATION
# ==========================================================================================

def load_input_data():
    """ This imports all content from the 'input.json' file. """
    # Define the input path
    script_dir = Path(__file__).parent
    input_path = script_dir/"W1_input.json"
    # Open and load JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)

    return data

# ==========================================================================================
#                                    AIRFOIL FUNCTIONS
# ==========================================================================================

def elliptic(data,zpb,pi):
    """ This generates the desired LL for an elliptic wing. """
    # Find the aspect ratio
    RA = data["wing"]["planform"]["aspect_ratio"]
    # Find the chord distribution
    cpb = (4/(pi*RA)) * np.sqrt(1-(2*zpb)**2)

def geometry(data,zpb,pi):
    """ This is the main function to develope the shape of the wing. """
    # Decide how the geometry should be directed
    planform = data["wing"]["planform"]["type"]
    match planform:
        case "elliptic":
            elliptic(data,zpb,pi)
        case "tapered":
            tapered(data,zpb,pi)
        case _:
            raise ValueError(f"Unknown wing planform: {planform}")

def tapered(data,zpb,pi):
    """ This generates the desired LL for a tapered wing. """

# ==========================================================================================
#                                   CALCULATED RESULTS
# ==========================================================================================

def fourier():
    """ This is where the fourier series coefficients are found. """
    

def results(data):
    """ This is where the final results are found. """

# ==========================================================================================
#                                        PLOTTING
# ==========================================================================================

def PlotLL(zpb):
    """ This displays the desired lifitng-line for a given wingspan. """
    fig, ax = plt.subplots()
    ax.scatter(zpb, np.zeros(len(zpb)))
    ax.set_xlabel("z/b")
    ax.set_yticks([])
    ax.plot()

    plt.show()

# ==========================================================================================
#                                     RUNNING PROGRAM
# ==========================================================================================

if __name__ == "__main__":
    # Constants
    pi = np.pi
    # Running the program
    main(pi)