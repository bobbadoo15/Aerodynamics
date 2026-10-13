'''
This program genearates a theoretical lifting line by using '...the numerical Fourier series 
solution to Prandtl's lifting-line theory to predict the aerodynamic characteristics of a 
finite wing with a given planform, washout, aileron deflection, and rigid-body roll.' The user 
can create a JSON file for a desired airfoil shape to use as the interested input file. 

Assumptions made for this project:
    * Thin Airfoil Theory: CLa = 2pi, al0 = 0
'''

# ==========================================================================================
#                                        IMPORTS
# ==========================================================================================

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================================================================
#                                          MAIN
# ==========================================================================================

def main(pi):
    """ This contains the structure of the program. """
    # Import JSON content from 'input.json'
    a, alo, CLa_tilde, nodes, planform, RA, RT = load_input_data()
    # Create the number of nodes along the semispan
    total_nodes = node(nodes)
    # Preallocate arrays
    B, c, cinv = preallocate(total_nodes)
    # Calculate for the theta values
    theta, zpb = cov(total_nodes,pi)
    # Define the shape of the wing
    an, c, cinv, cpb, CL, CLa, CDi, es, kL, kD = geometry(a,alo,B,CLa_tilde,c,cinv,planform,total_nodes,RA,RT,theta,zpb,pi)
    # Print, output results
    # results(an,c,cinv,cpb,CDi,CL,CLa,es,kD,kL,theta,zpb)
    # Plot the planform
    PlotPlanform(cpb,planform,zpb)

    # Debugging
    # np.set_printoptions(suppress=True, linewidth=200, 
    #                     formatter={'float_kind': '{:18.12f}'.format})
    # print(c)
    # print()
    # print(cinv)
    # print()
    # print(f"CL = {CL}")
    # print(f"CLa = {CLa}")
    # print(f"CDi = {CDi}")
    # print(f"kL = {kL}")

# ==========================================================================================
#                                       NODAL COUNT
# ==========================================================================================

def node(nodes):
    """ This function determines the node count the user desires. """
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
    # ====== Store data values ======
    # Define alpha in radians
    a = data["condition"]["alpha_root[deg]"] * (pi/180)
    # Define zero-lift alpha
    alo = 0 # ---> we are assuming 0 due to TAT, replace when told otherwise
    # Define section lift slope
    CLa_tilde = data["wing"]["airfoil_lift_slope"]
    # Import the total number of desired nodes
    nodes = data["wing"]["nodes_per_semispan"]
    # Decide how the geometry should be directed
    planform = data["wing"]["planform"]["type"]
    # Find the aspect ratio
    RA = data["wing"]["planform"]["aspect_ratio"]
    # Find the tapered ratio
    RT = data["wing"]["planform"]["taper_ratio"]
    return a, alo, CLa_tilde, nodes, planform, RA, RT

# ==========================================================================================
#                                    AIRFOIL FUNCTIONS
# ==========================================================================================

def elliptic(a,alo,B,CLa_tilde,c,cinv,N,RA,theta,zpb,pi):
    """ This generates the desired LL for an elliptic wing. """
    # Find the chord distribution
    cpb = (4/(pi*RA)) * np.sqrt(1-(2*zpb)**2)
    # Build the c-matrix
    c = cmatrix(c,cpb,CLa_tilde,N,theta)
    # Build the inverse c-matrix
    cinv = cmatrix_inverse(c,cinv)
    # Find Fourier values
    an = fourier(B,c)
    # Define induced drag and lift factors
    kL = 0
    kD = 0
    es = None
    # Define lift slope
    CLa = CLa_tilde / (1 + (CLa_tilde/(pi*RA)))
    # Find lift coefficient
    CL = CLa*(a-alo)
    # Find induced drag
    CDi = (CL**2) / (pi*RA)
    return an, c, cinv, cpb, CL, CLa, CDi, es, kL, kD

def geometry(a,alo,B,CLa_tilde,c,cinv,planform,N,RA,RT,theta,zpb,pi):
    """ This is the main function to develope the shape of the wing. """
    match planform:
        case "elliptic":
            an, c, cinv, cpb, CL, CLa, CDi, es, kL, kD = elliptic(a,alo,B,CLa_tilde,c,cinv,N,RA,theta,zpb,pi)
        case "tapered":
            an, c, cinv, cpb, CL, CLa, CDi, es, kL, kD = tapered(a,alo,B,CLa_tilde,c,cinv,N,RA,RT,theta,zpb,pi)
        case _:
            raise ValueError(f"Unknown wing planform: {planform}")

    return an, c, cinv, cpb, CL, CLa, CDi, es, kL, kD

def tapered(a,alo,B,CLa_tilde,c,cinv,N,RA,RT,theta,zpb,pi):
    """ This generates the desired LL for a tapered wing. """
    # Find the chord distribution
    cpb = (2 / (RA*(1+RT))) * (1 - (1-RT)*np.abs(2*zpb))
    # Build the c-matrix
    c = cmatrix(c,cpb,CLa_tilde,N,theta)
    # Build the inverse c-matrix
    cinv = cmatrix_inverse(c,cinv)
    # Find Fourier values
    an = fourier(B,c)
    # Find the first Fourier value
    a1 = an[0]
    # Find the factor values
    kL, kD, es = factors(an,a1,CLa_tilde,RA)
    # Define lift slope
    CLa = CLa_tilde / ((1 + (CLa_tilde/(pi*RA)))*(1+kL))
    # Find lift coefficient
    CL = CLa*(a-alo)
    # Find induced drag
    CDi = (CL**2) / (pi*RA*es)

    return an, c, cinv, cpb, CL, CLa, CDi, es, kL, kD

# ==========================================================================================
#                                   CALCULATED RESULTS
# ==========================================================================================

def cmatrix(c,cpb,CLa_tilde,N,theta):
    """ This is where the c-matrix is built. """
    for i in range(N):
        for j in range(N):
            n = j+1 # Fourier harmonic value; "j" value in MOF examples
            # Build matrix
            if i == 0:
                c[i,j] = n**2 # First-row elements
            elif i == N-1:
                c[i,j] = (-1)**(n+1) * n**2 # Last-row elements
            else:
                c[i,j] = ((4/(CLa_tilde*cpb[i])) + (n/np.sin(theta[i]))) * np.sin(n*theta[i]) # Elements in between
    return c

def cmatrix_inverse(c,cinv):
    """ This is where the c-matrix inverse is built. """
    cinv = np.linalg.inv(c)
    return cinv

def factors(an,a1,CLa_tilde,RA):
    """ This is where the lift slope and induced drag factors are found. """
    # Find lift slope factor
    kL = (1 - (1 + (pi*RA)/CLa_tilde)*a1) / ((1 + (pi*RA)/CLa_tilde)*a1)
    # Find the induced drag factor
    kD = 0
    for i in range(1,len(an)):
        n = i + 1
        kD += n * (an[i]/a1)**2
    # Find the span efficiency factor
    es = 1 / (1+kD)
    return kL, kD, es

def fourier(B,c):
    """ This is where the fourier series coefficients are found. """
    # Solve the Fourier Series
    an = np.linalg.solve(c,B)
    return an

def preallocate(N):
    """ This is where to preallocate matrices. """
    c = np.zeros((N,N))
    cinv = np.zeros((N,N))
    B = np.ones(N)
    return B, c, cinv
    
def results(an,c,cinv,cpb,CDi,CL,CLa,es,kD,kL,theta,zpb):
    """ This is where the final results are found. """
    # ========= Write out to a txt file =========
    script_dir = Path(__file__).parent
    output_path = script_dir/"Solution.txt"
    f = open(output_path,"w")
    # Theta
    f.write("\n================ THETA ===============\n")
    np.savetxt(f, theta, fmt="%18.12f")
    # z/b along the wingspan
    f.write("\n================= Z/B ================\n")
    np.savetxt(f, zpb, fmt="%18.12f")
    # c/b along the wingspan
    f.write("\n================= C/B ================\n")
    np.savetxt(f, cpb, fmt="%18.12f")
    # Matrices
    f.write("\n============== C-MATRIX ==============\n")
    np.savetxt(f, c, fmt="%18.12f")
    f.write("\n========== C-MATRIX INVERSE ==========\n")
    np.savetxt(f, cinv, fmt="%18.12f")
    f.write("\n======== LIFT SLOPE FACTOR, kL =======\n")
    np.savetxt(f, [kL], fmt="%18.12f")
    f.write("\n======= INDUCED DRAG FACTOR, kD ======\n")
    np.savetxt(f, [kD], fmt="%18.12f")
    f.write("\n====== SPAN EFFICIENCY FACTOR, es ====\n")
    np.savetxt(f, [es], fmt="%18.12f")
    f.write("\n========== LIFT SLOPE, CLa ===========\n")
    np.savetxt(f,[CLa], fmt="%18.12f")
    f.write("\n======== LIFT COEFFICIENT, CL ========\n")
    np.savetxt(f, [CL], fmt="%18.12f")
    f.write("\n========== INDUCED DRAG, CDi =========\n")
    np.savetxt(f, [CDi], fmt="%18.12f")
    f.write("\n========== FOURIER SERIES, an ========\n")
    np.savetxt(f, an, fmt="%18.12f")
    f.close()

# ==========================================================================================
#                                        PLOTTING
# ==========================================================================================

def PlotPlanform(cpb,planform,zpb):
    """ This displays the desired lifitng-line for a given wingspan. """
    # Define the LE and TE
    LE = 0.25 * cpb
    TE = -0.75 * cpb
    # Create the font family for the plot
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    })
    # Create figure
    fig, ax = plt.subplots()
    # Create plot information
    ax.set_title(f"{planform.capitalize()} Planform",
             fontsize=16,
             fontfamily="serif",
             fontweight="bold")
    ax.set_xlabel(r"$\mathbfit{z/b}$", fontsize=14)
    ax.set_ylabel(r"$\mathbfit{c/b}$", fontsize=14) 
    ax.set_aspect("equal")
    ax.grid(False)
    # Format plot borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Format ticks
    ax.tick_params(axis="both",
                which="major",
                width=1.5,
                length=6,
                direction="out")
    # Format tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    # Plot vertical chord lines
    ax.vlines(zpb, TE, LE, color="blue", linewidth=0.75)
    # Plot the LE and TE
    ax.plot(zpb, LE, color="black", linewidth=0.75)
    ax.plot(zpb, TE, color="black", linewidth=0.75)
    # Plot quarter chord line
    ax.plot(zpb, np.zeros_like(zpb), color="black", linestyle="--", linewidth=0.75)

    plt.show()

# ==========================================================================================
#                                     RUNNING PROGRAM
# ==========================================================================================

if __name__ == "__main__":
    # Constants
    pi = np.pi
    # Running the program
    main(pi)