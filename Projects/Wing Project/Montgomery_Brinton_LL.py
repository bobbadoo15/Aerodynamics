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

# ==========================================================================================
#                                          MAIN
# ==========================================================================================

def main(pi):
    """ This contains the structure of the program. """
    # Import Greek letters for displaying
    greek = Greek()
    # Import JSON content from 'input.json'
    a,alo,CLa_tilde,nodes,planform,RA,RT,view_planform,view_washout_dist,view_idrag,\
        Omega_value,washout_type,CLd,view_aileron,zr,zt,cfpc_begin,cfpc_end,\
            eta_h,delta_a,pbar,planform_file = load_input_data()
    # Create the number of nodes along the semispan
    total_nodes = node(nodes)
    # Calculate for the theta values
    theta,zpb = cov(total_nodes,pi)
    # Determine the cpb
    croot = None
    match planform:
        case "file":
            cpb,croot,RA = import_planform(planform_file,zpb)
            print(RA)
        case "elliptic" | "tapered":
            cpb = chord_distribution(planform,RA,RT,zpb,pi)
        case _:
            raise ValueError(f"Unknown planform type: {planform}")
    # Preallocate arrays
    B,C,Cinv = preallocate(total_nodes)
    # Define the Washout of the wing
    omega_dist,Omega_opt = washout(cpb,CLa_tilde,CLd,Omega_value,planform,RT,washout_type,zpb,croot)
    # Define the shape of the wing
    an,bn,C,Cinv,cpb,CL,CLa,CDi,dn,eps_Omega,es,kL,kD,kDL,kDOmega,omega_dist,Omega_opt\
    = geometry(a,alo,B,cpb,CLa_tilde,CLd,C,Cinv,planform,total_nodes,omega_dist,Omega_opt,RA\
               ,theta,washout_type,pi)
    # Calculate for aileron effects
    cr,ct,cfpc,chi,cn,CDI,Clda,Clpbar,Cl,Cn,pbar,yr,yt = aileron(a,an,alo,bn,C,CL,cfpc_begin,cfpc_end,cpb,delta_a,dn,\
                                         eta_h,pbar,Omega_opt,planform,planform_file,RA,RT,total_nodes,zpb,zr,zt)
    # Print, output results
    results(an,bn,cn,CDI,Cl,Clda,Clpbar,C,Cinv,cpb,CDi,CL,Cn,CLa,dn,eps_Omega,es,greek,kD,kDL,\
            kDOmega,kL,omega_dist,pbar,theta,zpb)
    # Plot the planform
    PlotPlanform(cr,ct,cfpc,cpb,greek,planform,view_aileron,view_planform,yr,yt,zpb,zr,zt)
    # Plot the washout distribution
    PlotWashout(greek,omega_dist,view_washout_dist,washout_type,zpb)
    # Plot the CDi
    PlotCDi(alo,CLa,eps_Omega,kD,kDL,kDOmega,Omega_opt,planform,RA,view_idrag,washout_type)
    # plot the aileron distribuiton
    PlotAileron(chi,greek,planform,view_aileron,zpb)

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

def import_planform(planform_file,zpb):
    """Import and interpolate the chord distribution."""
    # Find the file's path in the directory
    script_dir = Path(__file__).parent
    file_path = script_dir / planform_file
    # Load the data without the headings
    data = np.loadtxt(file_path, skiprows=1)
    # Define variables for each column
    zpb_input = data[:,0]
    cpb_input = data[:,1]
    # Validate the imported data
    if np.any(np.diff(zpb_input) <= 0):
        raise ValueError("z/b values must be strictly increasing.")
    if zpb_input[0] != 0 or zpb_input[-1] < 0.5:
        raise ValueError("Planform must cover the entire semispan.")
    if np.any(cpb_input < 0):
        raise ValueError("Chord lengths cannot be negative.")
    # Find the interpolated cpb values from node-defined zpb values
    cpb = np.interp(np.abs(zpb), zpb_input, cpb_input)
    # Find the average chord length
    cavg = np.trapezoid(cpb,zpb)
    # Find the related aspect ratio
    RA = 1.0/ (cavg)
    # Avoid a zero chord length
    cpb = np.maximum(cpb, 0.001)
    # Define the root of the chord
    croot = cpb[np.isclose(zpb,0.0)][0]
    return cpb,croot,RA

def load_input_data():
    """ This imports all content from the 'input.json' file. """
    # Define the input path
    script_dir = Path(__file__).parent
    input_path = script_dir/"input.json"
    # Open and load JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)
    # ====== Store data values ======
    # Define alpha in radians
    a = np.deg2rad(float(data["condition"]["alpha_root[deg]"]))
    # Define zero-lift alpha
    alo = 0.0 # ---> we are assuming 0 due to TAT, replace when told otherwise
    # Define section lift slope
    CLa_tilde = float(data["wing"]["airfoil_lift_slope"])
    # Import the total number of desired nodes
    nodes = int(data["wing"]["nodes_per_semispan"])
    # Decide how the geometry should be directed
    planform = data["wing"]["planform"]["type"]
    # Find the aspect ratio
    RA = float(data["wing"]["planform"].get("aspect_ratio", 0.0))
    # Find the tapered ratio
    RT = float(data["wing"]["planform"].get("taper_ratio", 0.0))
    # Load viewing options
    view_planform = data["view"]["planform"]
    view_washout_dist = data["view"]["washout_distribution"]
    view_aileron = data["view"]["aileron_distribution"]
    view_idrag = data["view"].get("induced_drag", False)
    # Load washout amount
    Omega_value = data["wing"]["washout"]["amount[deg]"]
    # Convert to radians unless calculating optimum washout
    if Omega_value != "optimum":
        Omega_value = float(np.deg2rad(Omega_value))
    # Load washout distribution type
    washout_type = data["wing"]["washout"]["distribution"]
    # Load the lift coefficient from design
    CLd = float(data["wing"]["washout"]["CL_design"])
    # Import aileron effects
    zr = float(data["wing"]["aileron"]["begin[z/b]"])
    zt = float(data["wing"]["aileron"]["end[z/b]"])
    cfpc_begin = float(data["wing"]["aileron"]["begin[cf/c]"])
    cfpc_end = float(data["wing"]["aileron"]["end[cf/c]"])
    eta_h = float(data["wing"]["aileron"]["hinge_efficiency"])
    delta_a = float(np.deg2rad(data["condition"]["aileron_deflection[deg]"]))
    pbar = data["condition"]["pbar"]
    if pbar != "steady":
        pbar = float(pbar)
    # Import planform text file if present in JSON input
    planform_file = data["wing"]["planform"].get("filename", None)
    return a,alo,CLa_tilde,nodes,planform,RA,RT,view_planform, view_washout_dist,view_idrag,\
        Omega_value,washout_type,CLd,view_aileron,zr,zt,cfpc_begin,cfpc_end,eta_h,\
        delta_a,pbar,planform_file

# ==========================================================================================
#                                    AIRFOIL FUNCTIONS
# ==========================================================================================

def aileron(a,an,al0,bn,C,CL,cfpc_begin,cfpc_end,cpb,delta_a,dn,eta_h,pbar,Omega,planform,planform_file,RA,RT,N,zpb,zr,zt):
    """ This calculates for the aileron effects. """
    # Define the section flap deflection efficiency
    eta_d = 1.0
    # Find the chord flap fraction
    cfpc,cr,ct,yr,yt = flap_chord_fraction(zpb,cpb,zr,zt,cfpc_begin,cfpc_end,planform,planform_file,RA,RT)
    # Find the local section flap effectiveness
    eps_f,eps_fi = flap_effectiveness(cfpc,eta_h,eta_d)
    # Find the control surface distribution
    chi = control_surface_distrubtion(eps_f,zpb,zr,zt)
    # Find the decomposed fourier 'c'
    cn = fourier(None,C,None,None,None,chi=chi)
    # Find the rolling coefficients
    An,Clda,Clpbar,Cl,pbar = rolling(a,al0,an,bn,cn,delta_a,dn,Omega,pbar,RA)
    # Find the induced drag due to aileron effects
    A2 = An[1]
    A3 = An[2]
    CDI_sum = 0.0
    for i in range(N):
        CDI_sum += (i+1) * An[i]**2
    CDI = (pi*RA*CDI_sum) - (0.5*pi*RA*pbar*A2)
    # Find the yawing moment coefficient for a general planform and twist
    Cn_sum = 0.0
    for i in range(4,N+1):
        Cn_sum += (2*i-1) * An[i-2]*An[i-1]
    Cn = ((CL*((6*A2)-pbar))/8) + ((pi*RA)*A3*((10*A2)-pbar)/8) + ((pi*RA/4)*Cn_sum)
    return cr,ct,cfpc,chi,cn,CDI,Clda,Clpbar,Cl,Cn,pbar,yr,yt
    
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

def control_surface_distrubtion(eps_f,z,zr,zt):
    """ This finds the control surface distribution, χ (chi). """
    # Allocate space for chi
    chi = np.zeros_like(z,dtype=float)
    # Loop through each node in the piecewise function
    for i in range(len(z)):
        # Left aileron wing
        if zr <= z[i] <= zt:
            chi[i] = -eps_f[i]
        # Right aileron wing
        elif -zt <= z[i] <= -zr:
            chi[i] = eps_f[i]
        # Not within aileron range
        else:
            chi[i] = 0.0
    # print(chi)
    return chi

def flap_chord_fraction(zpb,cpb,zr,zt,cfpc_begin,cfpc_end,planform,planform_file,RA,RT):
    if planform == "file":
        data = np.loadtxt(Path(__file__).parent / planform_file, skiprows=1)
        cr = np.interp(zr, data[:,0], data[:,1])
        ct = np.interp(zt, data[:,0], data[:,1])
    else:
        cr = chord_distribution(planform, RA, RT, zr, pi)
        ct = chord_distribution(planform, RA, RT, zt, pi)
    # Find hinge-line positions
    yr = (cfpc_begin - 0.75)*cr
    yt = (cfpc_end - 0.75)*ct
    # Initialize array
    cfpc = np.zeros_like(zpb, dtype=float)
    # Checks if the hinge lines are withing the planform
    mask = (np.abs(zpb) >= zr) & (np.abs(zpb) <= zt)
    if np.any((cfpc[mask] < 0) | (cfpc[mask] > 1)):
        raise ValueError("Aileron hinge line extends outside the wing.")
    # Loop through each spanwise node
    for i in range(len(zpb)):
        z = abs(zpb[i])
        # Check whether node is within aileron
        if zr <= z <= zt:
            # Find hinge-line position at this node
            y = yr + (yt-yr)*(z-zr)/(zt-zr)
            # Calculate local flap fraction
            cfpc[i] = y/cpb[i] + 0.75
    return cfpc,cr,ct,yr,yt

def flap_effectiveness(cfpc,eta_h,eta_d):
    """ This finds the flap effectiveness. """
    # Find the change in variables
    theta_f = np.acos(2.0*cfpc - 1.0)
    # Find the ideal section flap effectiveness
    eps_fi = 1.0 - ((theta_f - np.sin(theta_f))/pi)
    # Find the actual section flap effectiveness
    eps_f = eta_h*eta_d*eps_fi
    return eps_f,eps_fi

def geometry(a,alo,B,cpb,CLa_tilde,CLd,C,Cinv,planform,N,omega_dist,Omega_opt,RA,theta,
             washout_type,pi):
    """Calculate aerodynamic properties of the wing."""
    # Build C-matrix and its inverse
    C = cmatrix(C, cpb, CLa_tilde, N, theta)
    Cinv = cmatrix_inverse(C, Cinv)
    # Calculate Fourier coefficients
    an,bn,dn = fourier(B,C,omega_dist,theta,washout_type,chi=None)
    # First Fourier coefficients
    a1 = an[0]
    b1 = bn[0] if bn is not None else None
    # Calculate aerodynamic factors
    eps_Omega,kL,kD,kDL,kDOmega,es = factors(an,a1,bn,b1,CLa_tilde,planform,RA)
    # Calculate lift slope
    match planform:
        case "elliptic":
            CLa = CLa_tilde / (1 + CLa_tilde/(pi*RA))
        case "tapered" | "file":
            CLa = CLa_tilde / ((1 + CLa_tilde/(pi*RA))*(1+kL))
            if planform == "file":
                # Define the optimum washout for txt file
                if washout_type == "optimum" and Omega_opt == "optimum":
                    if np.isclose(kDOmega, 0.0):
                        raise ValueError("Cannot optimize washout: kDOmega is zero.")
                    Omega_opt = (kDL*CLd)/(2*kDOmega*CLa)
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
    return (an,bn,C,Cinv,cpb,CL,CLa,CDi,dn,eps_Omega,es,kL,kD,kDL,kDOmega,omega_dist,Omega_opt)

def rolling(a,al0,an,bn,cn,delta_a,dn,Omega,pbar,RA):
    """ This finds everything related to rolling. """
    # Find the specific decomposed fourier values
    c2 = cn[1]
    d2 = dn[1]
    # Find the change in rolling moment coefficient with respect to aileron deflection
    Clda = -0.25*pi*RA*c2
    # Find the rolling moment coefficient with respect to dimensionless rolling rate
    Clpbar = -0.25*pi*RA*d2
    # Define steady pbar if present
    if pbar == 'steady':
        pbar = (-Clda*delta_a)/(Clpbar)
        Cl = 0.0
    else:
        # Find the rolling moment coefficient
        Cl = (Clda*delta_a) + (Clpbar*pbar)
    # Find the Fourier Coefficient An
    An = (an*(a-al0)) + (cn*delta_a) + (dn*pbar)
    if bn is not None:
        An -= bn*Omega
    return An,Clda,Clpbar,Cl,pbar

def washout(cpb,CLa_tilde,CLd,Omega_value,planform,RT,washout_type,zpb,croot):
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
                case "tapered" | "file":
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
            omega_dist,Omega_opt = washout_optimum_distribution(cpb,CLa_tilde,CLd,Omega_value,planform,RT,zpb,croot)
        case _:
            raise ValueError(f"Unknown washout type: {washout_type}")
    return omega_dist,Omega_opt

def washout_optimum_distribution(cpb,CLa_tilde,CLd,Omega_value,planform,RT,zpb,croot):
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
        case "file":
            omega_dist = 1 - np.sqrt(1-(2*zpb)**2)/(cpb/croot)
            match Omega_value:
                case int() | float():
                    Omega_opt = Omega_value
                case "optimum":
                    Omega_opt = "optimum"
                case _:
                    raise ValueError(f"Unknown washout amount: {Omega_value}")
        case _:
            raise ValueError(f"Unknown planform type: {planform}")
    return omega_dist, Omega_opt

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
        case "tapered" | "file":
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

def fourier(B,C,omega_dist,theta,washout_type,chi=None):
    """ This is where the fourier series coefficients are found. """
    # Find the c-values if chi is present
    if chi is not None:
        cn = np.linalg.solve(C, chi)
        return cn
    # Solve the Fourier Series
    an = np.linalg.solve(C,B)
    match washout_type:
        case "none":
            bn = None
        case "linear" | "optimum":
            bn = np.linalg.solve(C,omega_dist)
        case _:
            raise ValueError(f"Unknown washout type: {washout_type}")
    Cos = np.cos(theta)
    dn = np.linalg.solve(C,Cos)
    return an,bn,dn

def preallocate(N):
    """ This is where to preallocate matrices. """
    C = np.zeros((N,N))
    Cinv = np.zeros((N,N))
    B = np.ones(N)
    return B, C, Cinv
    
def results(an,bn,cn,CDI,Cl,Clda,Clpbar,C,Cinv,cpb,CDi,CL,Cn,CLa,dn,eps_Omega,es,greek,kD,kDL,kDOmega,kL,omega_dist,pbar,theta,zpb):
    """ This is where the final results are found. """
    # ========= Write out to a txt file =========
    script_dir = Path(__file__).parent
    output_path = script_dir/"Solution.txt"
    f = open(output_path,"w")
    f.write("\n============== C-MATRIX ==============\n")
    np.savetxt(f, C, fmt="%18.12f")
    f.write("\n========== C-MATRIX INVERSE ==========\n")
    np.savetxt(f, Cinv, fmt="%18.12f")
    f.write("\n========== FOURIER SERIES, an ========\n")
    np.savetxt(f, an, fmt="%18.12f")
    f.write("\n========== FOURIER SERIES, bn ========\n")
    if bn is not None:
        np.savetxt(f, bn, fmt="%18.12f")
    else:
        f.write("No washout applied.\n")
    f.write("\n========== FOURIER SERIES, cn ========\n")
    np.savetxt(f, cn, fmt="%18.12f")
    f.write("\n========== FOURIER SERIES, dn ========\n")
    np.savetxt(f, dn, fmt="%18.12f")
    f.close()
    # Print the solutions to the screen
    print()
    print("==============================================")
    print("            AERODYNAMIC PROPERTIES")
    print("==============================================")
    print(f"{greek['kappa']['lower']}L      ={kL:18.12f}")
    print(f"{greek['kappa']['lower']}D      ={kD:18.12f}")
    print(f"es      ={es:18.12f}")
    print(f"CL,{greek['alpha']['lower']}    ={CLa:18.12f}")
    print(f"{greek['epsilon']['lower']}{greek['omega']['upper']}      ={eps_Omega:18.12f}")
    print(f"{greek['kappa']['lower']}DL     ={kDL:18.12f}")
    print(f"{greek['kappa']['lower']}D{greek['omega']['upper']}     ={kDOmega:18.12f}")
    print(f"Cl,{greek['delta']['lower']}a   ={Clda:18.12f}")
    print(f"Cl,pbar ={Clpbar:18.12f}")
    print(f"pbar    ={pbar:18.12f}")
    print(f"CL      ={CL:18.12f}")
    print(f"CDi     ={CDi:18.12f}")
    print(f"CDI     ={CDI:18.12f}")
    print(f"Cl      ={Cl:18.12f}")
    print(f"Cn      ={Cn:18.12f}")
    print("==============================================\n")

# ==========================================================================================
#                                        PLOTTING
# ==========================================================================================

def PlotAileron(chi,greek,planform,view_aileron,z):
    """ This plots the aileron distribution. """
    if view_aileron == True:
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
        ax.set_title(f"{planform.capitalize()} Planform Aileron Distribution",
                fontsize=16,
                fontfamily="serif",
                fontweight="bold")
        ax.set_xlabel(r"$\mathbfit{z/b}$", fontsize=14)
        ax.set_ylabel(rf"$\mathbfit{{{greek['chi']['lower']}}} (z/b)$", fontsize=14)        
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
        # Plot the distribution
        ax.plot(z,chi,color="black",linewidth=0.75)
        plt.show()
    else:
        print("Displaying aileron distribution view is disabled.")

def PlotCDi(alo,CLa,eps_Omega,kD,kDL,kDOmega,Omega,planform,RA,view_idrag,washout_type):
    """ Plot CDi vs CL for no twist and linear twist. """
    if view_idrag == True:
        # Create an array of alphas
        alpha_deg = np.linspace(-12,12,100)
        alpha_rad = np.deg2rad(alpha_deg)
        # Calculate CL without twist
        CL_none = CLa*(alpha_rad-alo)
        # Calculate CL with linear twist
        CL_linear = CLa*((alpha_rad-alo)-(eps_Omega*Omega))
        # Calculate CDi without twist
        CDi_none = (CL_none**2*(1+kD))/(np.pi*RA)
        # Calculate CDi with linear twist
        CDi_linear = (
            CL_linear**2*(1+kD)
            - kDL*CL_linear*CLa*Omega
            + kDOmega*(CLa*Omega)**2
        )/(np.pi*RA)
        # Store the arrays
        script_dir = Path(__file__).parent
        output_path = script_dir/"CL_CDi_comparison.txt"
        data = np.column_stack((
            alpha_deg,CL_none,CDi_none,CL_linear,CDi_linear
        ))
        np.savetxt(
            output_path,
            data,
            header="alpha[deg] CL_none CDi_none CL_linear CDi_linear",
            fmt="%.12f"
        )
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
        # Plot no twist
        ax.plot(CL_none,CDi_none,
                color="black",
                linewidth=2.5,
                label="No Twist")
        # Plot linear twist
        ax.plot(CL_linear,CDi_linear,
                color="orange",
                linestyle="--",
                linewidth=2.0,
                label=f"{np.rad2deg(Omega):.1f}° Linear Twist")
        # Format plot
        ax.set_title(
            rf"{planform.capitalize()} Wing $C_{{D_i}}$ Comparison",
            fontsize=16,
            fontfamily="serif",
            fontweight="bold"
        )
        ax.set_xlabel(r"$C_L$",fontsize=14)
        ax.set_ylabel(r"$C_{D_i}$",fontsize=14)
        ax.grid(True)
        ax.legend()
        # Format plot borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # Format ticks
        ax.tick_params(
            axis="both",
            which="major",
            width=1.5,
            length=6,
            direction="out"
        )
        plt.show()
    else:
        print("Displaying CDi view is disabled.")

def PlotPlanform(cr,ct,cfpc,cpb,greek,planform,view_aileron,view_planform,yr,yt,zpb,zr,zt):
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
        # Plot ailerons if they exist
        if view_aileron == True:
            # Find the TE positions for the relative cf/c
            TEr = -0.75*cr
            TEt = -0.75*ct
            # Find nodes within the ailerons
            mask = (np.abs(zpb) >= zr) & (np.abs(zpb) <= zt)
            # Find hinge-line positions at the nodes
            hinge = TE + cfpc*cpb
            # Plot red flap chord lines
            ax.vlines(zpb[mask], TE[mask], hinge[mask],
                    color="red", linewidth=0.75)
            # Plot right aileron
            ax.plot([zr, zt], [yr, yt], color="black", linewidth=1.5)
            ax.vlines([zr, zt], [TEr, TEt], [yr, yt],
                    color="black", linewidth=1.5)
            # Plot left aileron
            ax.plot([-zt, -zr], [yt, yr], color="black", linewidth=1.5)
            ax.vlines([-zt, -zr], [TEt, TEr], [yt, yr],
                    color="black", linewidth=1.5)
            # Shade aileron regions
            # ax.fill([zr, zt, zt, zr],
            #         [yr, yt, TEt, TEr],
            #         color="orange", alpha=0.4)
            # ax.fill([-zt, -zr, -zr, -zt],
            #         [yt, yr, TEr, TEt],
            #         color="orange", alpha=0.4)
        else:
            print("Displaying ailerons is disabled.")
        # Plot the LE and TE
        ax.plot(zpb, LE, color="black", linewidth=1.5)
        ax.plot(zpb, TE, color="black", linewidth=1.5)
        # Plot the tapered edges
        if planform == "tapered" or (planform == "file" and cpb[0] > 0):
            # Plot the left wingtip
            ax.vlines(zpb[0], TE[0], LE[0], color="black", linewidth=1.5)
            # Plot the right wingtip
            ax.vlines(zpb[-1], TE[-1], LE[-1], color="black", linewidth=1.5)
        # Plot quarter chord line
        ax.plot(zpb, np.zeros_like(zpb), color="black", linestyle="--", linewidth=1.0)
        plt.show()
    else:
        print("Displaying planform view is disabled.")

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
        ax.set_ylabel(rf"$\mathbfit{{{greek['omega']['lower']}}} (z/b)$", fontsize=14)        
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
        print("Displaying the washout distribution view is disabled.")

# ==========================================================================================
#                                     RUNNING PROGRAM
# ==========================================================================================

if __name__ == "__main__":
    # Constants
    pi = np.pi
    # Running the program
    main(pi)