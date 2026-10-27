'''
This program genearates a theoretical lifting line by using '...the numerical Fourier series 
solution to Prandtl's lifting-line theory to predict the aerodynamic characteristics of a 
finite wing with a given planform, washout, aileron deflection, and rigid-body roll.' The user 
can create a JSON file for a desired airfoil shape to use as the interested input file. 

Assumptions made for this project:
    * Thin Airfoil Theory: CLa = 2pi, al0 = 0

GREEK UNICODE REFERENCE SHEET
Usage: print("\u03B1") will print α

--- LOWERCASE LETTERS ---
\u03B1 : α (alpha)      \u03B7 : η (eta)        \u03BD : ν (nu)         \u03C4 : τ (tau)
\u03B2 : β (beta)       \u03B8 : θ (theta)      \u03BE : ξ (xi)         \u03C5 : υ (upsilon)
\u03B3 : γ (gamma)      \u03B9 : ι (iota)       \u03BF : ο (omicron)    \u03C6 : φ (phi)
\u03B4 : δ (delta)      \u03BA : κ (kappa)      \u03C0 : π (pi)         \u03C7 : χ (chi)
\u03B5 : ε (epsilon)    \u03BB : λ (lambda)     \u03C1 : ρ (rho)        \u03C8 : ψ (psi)
\u03B6 : ζ (zeta)       \u03BC : μ (mu)         \u03C3 : σ (sigma)      \u03C9 : ω (omega)

--- UPPERCASE LETTERS ---
\u0391 : Α (Alpha)      \u0397 : Η (Eta)        \u039D : Ν (Nu)         \u03A4 : Τ (Tau)
\u0392 : Β (Beta)       \u0398 : Θ (Theta)      \u039E : Ξ (Xi)         \u03A5 : Υ (Upsilon)
\u0393 : Γ (Gamma)      \u0399 : Ι (Iota)       \u039F : Ο (Omicron)    \u03A6 : Φ (Phi)
\u0394 : Δ (Delta)      \u039A : Κ (Kappa)      \u03A0 : Π (Pi)         \u03A7 : Χ (Chi)
\u0395 : Ε (Epsilon)    \u039B : Λ (Lambda)     \u03A1 : Ρ (Rho)        \u03A8 : Ψ (Psi)
\u0396 : Ζ (Zeta)       \u039C : Μ (Mu)         \u03A3 : Σ (Sigma)      \u03A9 : Ω (Omega)
'''

# ==========================================================================================
#                                        IMPORTS
# ==========================================================================================

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# ==========================================================================================
#                                          MAIN
# ==========================================================================================

def main(pi):
    """ This contains the structure of the program. """
    # Import Greek letters for displaying
    greek = Greek()
    # Import JSON content from 'input.json'
    a,alo,CLa_tilde,nodes,\
    planform,RA,RT,view_planform,\
    view_washout_dist,Omega_value,\
    washout_type,CLd = load_input_data()
    # Create the number of nodes along the semispan
    total_nodes = node(nodes)
    # Preallocate arrays
    B,c,cinv = preallocate(total_nodes)
    # Calculate for the theta values
    theta,zpb = cov(total_nodes,pi)
    # Define the Washout of the wing
    omega_dist,Omega_opt = washout(CLa_tilde,CLd,Omega_value,planform,RT,washout_type,zpb)
    # Define the shape of the wing
    an,bn,c,cinv,cpb,CL,CLa,CDi,eps_Omega,\
    es,kL,kD,kDL,kDOmega,omega_dist = geometry(a,alo,B,CLa_tilde,CLd,c,cinv,planform,total_nodes,omega_dist,
                                    Omega_opt,RA,RT,theta,Omega_value,washout_type,zpb,pi)
    # Print, output results
    results(an,bn,c,cinv,cpb,CDi,CL,CLa,eps_Omega,es,greek,kD,kDL,kDOmega,kL,omega_dist,theta,zpb)
    # Plot the planform
    PlotPlanform(cpb,greek,planform,view_planform,zpb)
    # Plot the washout distribution
    PlotWashout(greek,omega_dist,view_washout_dist,washout_type,zpb)

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
#                                   IMPORTING INFORMATION
# ==========================================================================================

def Greek():
    """ This stores greek unicode values. """
    greek = {
    'alpha'  : {'lower': '\u03B1', 'upper': '\u0391'},
    'beta'   : {'lower': '\u03B2', 'upper': '\u0392'},
    'gamma'  : {'lower': '\u03B3', 'upper': '\u0393'},
    'delta'  : {'lower': '\u03B4', 'upper': '\u0394'},
    'epsilon': {'lower': '\u03B5', 'upper': '\u0395'},
    'zeta'   : {'lower': '\u03B6', 'upper': '\u0396'},
    'eta'    : {'lower': '\u03B7', 'upper': '\u0397'},
    'theta'  : {'lower': '\u03B8', 'upper': '\u0398'},
    'iota'   : {'lower': '\u03B9', 'upper': '\u0399'},
    'kappa'  : {'lower': '\u03BA', 'upper': '\u039A'},
    'lambda' : {'lower': '\u03BB', 'upper': '\u039B'},
    'mu'     : {'lower': '\u03BC', 'upper': '\u039C'},
    'nu'     : {'lower': '\u03BD', 'upper': '\u039D'},
    'xi'     : {'lower': '\u03BE', 'upper': '\u039E'},
    'omicron': {'lower': '\u03BF', 'upper': '\u039F'},
    'pi'     : {'lower': '\u03C0', 'upper': '\u03A0'},
    'rho'    : {'lower': '\u03C1', 'upper': '\u03A1'},
    'sigma'  : {'lower': '\u03C3', 'upper': '\u03A3'},  # Word-final lowercase variant is '\u03C2' (ς)
    'tau'    : {'lower': '\u03C4', 'upper': '\u03A4'},
    'upsilon': {'lower': '\u03C5', 'upper': '\u03A5'},
    'phi'    : {'lower': '\u03C6', 'upper': '\u03A6'},
    'chi'    : {'lower': '\u03C7', 'upper': '\u03A7'},
    'psi'    : {'lower': '\u03C8', 'upper': '\u03A8'},
    'omega'  : {'lower': '\u03C9', 'upper': '\u03A9'}
}
    return greek

def load_input_data():
    """ This imports all content from the 'input.json' file. """
    # Define the input path
    script_dir = Path(__file__).parent
    input_path = script_dir/"W2_input.json"
    # Open and load JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)
    # ====== Store data values ======
    # Define alpha in radians
    a = np.deg2rad(data["condition"]["alpha_root[deg]"])
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
    # Load viewing options
    view_planform = data["view"]["planform"]
    view_washout_dist = data["view"]["washout_distribution"]
    # Load washout amount
    Omega_value = data["wing"]["washout"]["amount[deg]"]
    # Convert to radians unless calculating optimum washout
    if Omega_value != "optimum":
        Omega_value = np.deg2rad(Omega_value)
    # Load washout distribution type
    washout_type = data["wing"]["washout"]["distribution"]
    # Load the lift coefficient from design
    CLd = data["wing"]["washout"]["CL_design"]
    return a, alo, CLa_tilde, nodes, planform, RA, RT, view_planform, view_washout_dist, \
           Omega_value, washout_type, CLd

# ==========================================================================================
#                                    AIRFOIL FUNCTIONS
# ==========================================================================================

def chord_distribution(planform,RA,RT,zpb,pi):
    """Calculate the normalized chord distribution."""
    match planform:
        case "elliptic":
            cpb = (4/(pi*RA)) * np.sqrt(1-(2*zpb)**2)
        case "tapered":
            cpb = (2/(RA*(1+RT))) * \
                  (1-(1-RT)*np.abs(2*zpb))
        case _:
            raise ValueError(f"Unknown planform: {planform}")
    return cpb

def geometry(a,alo,B,CLa_tilde,CLd,c,cinv,planform,N,omega_dist,Omega_opt,RA,RT,theta,Omega_value,
             washout_type, zpb, pi):
    """Calculate aerodynamic properties of the wing."""
    # Calculate chord distribution
    cpb = chord_distribution(planform, RA, RT, zpb, pi)
    # Build C-matrix and its inverse
    c = cmatrix(c, cpb, CLa_tilde, N, theta)
    cinv = cmatrix_inverse(c, cinv)
    # Calculate Fourier coefficients
    an, bn = fourier(B, c, omega_dist, washout_type)
    # First Fourier coefficients
    a1 = an[0]
    b1 = bn[0] if bn is not None else None
    # Calculate aerodynamic factors
    eps_Omega, kL, kD, kDL, kDOmega, es = factors(an, a1, bn, b1, CLa_tilde, planform, RA)
    # Calculate lift slope
    match planform:
        case "elliptic":
            CLa = CLa_tilde / (1 + CLa_tilde/(pi*RA))
        case "tapered":
            CLa = CLa_tilde / ((1 + CLa_tilde/(pi*RA))*(1+kL))
    # Calculate lift and induced drag
    match washout_type:
        case "none":
            omega_dist = np.zeros(N)
            CL  = CLa*(a-alo)
            CDi = CL**2*(1+kD)/(pi*RA)
        case "linear":
            if Omega_opt == "optimum":
                if np.isclose(kDOmega, 0.0):
                    raise ValueError("Cannot optimize linear washout: kDOmega is zero.")
                Omega_opt = (kDL*CLd)/(2*kDOmega*CLa)
            CL  = CLa*((a-alo)-(eps_Omega*Omega_opt))
            CDi = (CL**2*(1+kD) - kDL*CL*CLa*Omega_opt + kDOmega*(CLa*Omega_opt)**2)/(pi*RA)
        case "optimum":
            CL  = CLa*((a-alo)-(eps_Omega*Omega_opt))
            CDi = (CL**2*(1+kD) - kDL*CL*CLa*Omega_opt + kDOmega*(CLa*Omega_opt)**2)/(pi*RA)
        case _:
            raise ValueError(f"Unknown washout type: {washout_type}")
    return (an,bn,c,cinv,cpb,CL,CLa,CDi,eps_Omega,es,kL,kD,kDL,kDOmega,omega_dist)

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
                c[i,j] = ((4/(CLa_tilde*cpb[i])) + (n/np.sin(theta[i]))) \
                          * np.sin(n*theta[i]) # Elements in between
    return c

def cmatrix_inverse(c,cinv):
    """ This is where the c-matrix inverse is built. """
    cinv = np.linalg.inv(c)
    return cinv

def factors(an, a1, bn, b1, CLa_tilde, planform, RA):
    """Calculate lift slope and induced drag factors."""
    # Initialize factors
    eps_Omega = 0.0
    kL = 0.0
    kD = 0.0
    kDL = 0.0
    kDOmega = 0.0
    es = 1.0
    # Determine factors based on planform
    match planform:
        case "elliptic":
            kL = 0.0
            if bn is not None:
                eps_Omega = b1/a1
        case "tapered":
            kL = (1 - (1 + (np.pi*RA)/CLa_tilde)*a1) / \
                 ((1 + (np.pi*RA)/CLa_tilde)*a1)
            if bn is not None:
                eps_Omega = b1/a1
        case _:
            raise ValueError(f"Unknown planform: {planform}")
    # Calculate induced drag factor
    for i in range(1, len(an)):
        n = i + 1
        kD += n*(an[i]/a1)**2
    # Calculate span efficiency
    es = 1/(1+kD)
    # Calculate washout-related induced drag factors
    if bn is not None:
        for i in range(1, len(an)):
            n = i + 1
            a_ratio = an[i]/a1
            difference = bn[i]/a1 - eps_Omega*a_ratio
            kDL += 2*n*a_ratio*difference
            kDOmega += n*difference**2
    return (eps_Omega, kL, kD, kDL, kDOmega, es)

def fourier(B,c,omega_dist,washout_type):
    """ This is where the fourier series coefficients are found. """
    # Solve the Fourier Series
    an = np.linalg.solve(c,B)
    match washout_type:
        case "none":
            bn = None
        case "linear" | "optimum":
            bn = np.linalg.solve(c,omega_dist)
        case _:
            raise ValueError(f"Unknown washout type: {washout_type}")
    return an, bn

def preallocate(N):
    """ This is where to preallocate matrices. """
    c = np.zeros((N,N))
    cinv = np.zeros((N,N))
    B = np.ones(N)
    return B, c, cinv
    
def results(an,bn,c,cinv,cpb,CDi,CL,CLa,eps_Omega,es,greek,kD,kDL,kDOmega,kL,omega_dist,theta,zpb):
    """ This is where the final results are found. """
    # ========= Write out to a txt file =========
    script_dir = Path(__file__).parent
    output_path = script_dir/"Solution.txt"
    f = open(output_path,"w")
    f.write("\n============== C-MATRIX ==============\n")
    np.savetxt(f, c, fmt="%18.12f")
    f.write("\n========== C-MATRIX INVERSE ==========\n")
    np.savetxt(f, cinv, fmt="%18.12f")
    f.write("\n========== FOURIER SERIES, an ========\n")
    np.savetxt(f, an, fmt="%18.12f")
    f.write("\n========== FOURIER SERIES, bn ========\n")
    if bn is not None:
        np.savetxt(f, bn, fmt="%18.12f")
    else:
        f.write("No washout applied.\n")
    f.close()
    # Print the solutions to the screen
    print()
    print("==============================================")
    print("            AERODYNAMIC PROPERTIES")
    print("==============================================")
    print(f"{greek['kappa']['lower']}L    ={kL:18.12f}")
    print(f"{greek['kappa']['lower']}D    ={kD:18.12f}")
    print(f"{greek['epsilon']['lower']}s    ={es:18.12f}")
    print(f"CL{greek['alpha']['lower']}   ={CLa:18.12f}")
    print(f"{greek['epsilon']['lower']}{greek['omega']['upper']}    ={eps_Omega:18.12f}")
    print(f"{greek['kappa']['lower']}DL   ={kDL:18.12f}")
    print(f"{greek['kappa']['lower']}D{greek['omega']['upper']}   ={kDOmega:18.12f}")
    print(f"CL    ={CL:18.12f}")
    print(f"CDi   ={CDi:18.12f}")
    print("==============================================\n")

def washout(CLa_tilde,CLd,Omega_value,planform,RT,washout_type,zpb):
    """ This is where washout is calculated. """
    match washout_type:
        case "none":
            omega_dist = np.zeros_like(zpb)
            Omega_opt  = 0.0
        case "linear":
            omega_dist = np.abs(2*zpb)
            match planform:
                case "elliptic":
                    match Omega_value:
                        case int() | float():
                            Omega_opt = Omega_value
                        case "optimum":
                            Omega_opt = 0.0
                        case _:
                            raise ValueError(f"Unknown washout amount type: {Omega_value}")
                case "tapered":
                    match Omega_value:
                        case int() | float():
                            Omega_opt = Omega_value
                        case "optimum":
                            Omega_opt = "optimum"
                        case _:
                            raise ValueError(f"Unknown washout amount type: {Omega_value}")
                case _:
                    raise ValueError(f"Unknown planform type: {planform}")
        case "optimum":
            omega_dist,Omega_opt = washout_optimum_distribution(CLa_tilde,CLd,Omega_value,planform,RT,zpb)
        case _:
            raise ValueError(f"Unknown washout type: {washout_type}")
    return omega_dist,Omega_opt

def washout_optimum_distribution(CLa_tilde,CLd,Omega_value,planform,RT,zpb):
    """ This finds the optimum washout distribution. """
    # Select the case for the wing planform
    match planform:
        case "elliptic":
            omega_dist = np.zeros_like(zpb)
            Omega_opt = 0.0
        case "tapered":
            omega_dist = 1 - (np.sqrt(1-(2*zpb)**2)) / (1 - (1-RT)*np.abs(2*zpb))
            match Omega_value:
                case int() | float():
                    Omega_opt = Omega_value
                case "optimum":
                    Omega_opt = (2*(1+RT)*CLd) / (pi*CLa_tilde)
                case _:
                    raise ValueError(f"Unknown washout amount: {Omega_value}")
        case _:
            raise ValueError(f"Unknown planform type: {planform}")
    return omega_dist, Omega_opt

# ==========================================================================================
#                                        PLOTTING
# ==========================================================================================

def PlotPlanform(cpb,greek,planform,view_planform,zpb):
    """ This displays the desired lifitng-line for a given wingspan. """
    if view_planform == True:
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
        # Plot the tapered edges
        if planform == "tapered":
            # Plot the left wingtip
            ax.vlines(zpb[0], TE[0], LE[0], color="black", linewidth=0.75)
            # Plot the right wingtip
            ax.vlines(zpb[-1], TE[-1], LE[-1], color="black", linewidth=0.75)
        # Plot quarter chord line
        ax.plot(zpb, np.zeros_like(zpb), color="black", linestyle="--", linewidth=0.75)

        plt.show()
    else:
        print("Displaying planform view is disabled. Goodbye!")
        sys.exit()

def PlotWashout(greek,omega_dist,view_washout_dist,washout_type,zpb):
    """ This plots the washout distribution. """
    if view_washout_dist:
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
        match washout_type:
            case "none":
                ax.set_title("No Washout Distribution",
                        fontsize=16,
                        fontfamily="serif",
                        fontweight="bold")
            case "linear":
                ax.set_title(f"{washout_type.capitalize()} Washout Distribution",
                        fontsize=16,
                        fontfamily="serif",
                        fontweight="bold")
            case "optimum":
                ax.set_title(f"{washout_type.capitalize()} Washout Distribution",
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
        # Plot washout
        ax.plot(zpb, omega_dist, color="black", linewidth=0.75)

        plt.show()
    else:
        print("Displaying planform view is disabled. Goodbye!")
        sys.exit()

# ==========================================================================================
#                                     RUNNING PROGRAM
# ==========================================================================================

if __name__ == "__main__":
    # Constants
    pi = np.pi
    # Running the program
    main(pi)