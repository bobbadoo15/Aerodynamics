"""
    This program uses the vortex-panel method outlined in Section 1.6 from Mechanics of Flight by Warren Phillips
    to compute the lift coefficient, leading-edge pitching-moment coefficient, and quarter-chord pitching-moment 
    coefficient at a specified angle of attack on an airfoil defined by a set of points that are read in from a 
    text file. 

    Note: This method assumes inviscid, incompressible, irrotational flow outside the vortex sheet. That means
    we can assume that there is no drag. We are also assuming the chord length is 1.
"""

import json
import csv
import numpy as np
from pathlib import Path

def main():
    """ This is the main function to run the program. """
    # Import data from the JSON file
    a, g, v = load_json_data()
    # Import data from the geometry
    nodes = load_geometry_data(g)
    # Find the relative nodes and panels for the given airfoil
    N, Np = nodes_and_panels(nodes)
    # Preallocate variables
    A,B,cp,gamma,l,P = preallocate(N,Np)
    # Find the panel lengths
    lengths = panel_length(l,nodes,Np)
    # Find the control points
    control_points = get_control_points(cp,nodes,Np)
    # Find the A-matrix
    A = airfoil_matrix(cp,lengths,nodes,Np,P,A)
    # Find the B-matrix
    B = bmatrix(a,B,lengths,Np,nodes,v)
    # Find the nodal vortex strengths
    gamma = nodal_vortex_strength(A,B,gamma)
    # Find the velocity components
    V = velocity(a,cp,v,gamma,nodes,lengths,Np)
    # Find the pressure coefficient
    Cp = pressure(V,v)
    # Find the lift coefficient
    CL = lift(gamma,lengths,Np,v)
    # Find the LE Pitching Moment Coefficient
    Cmle = le_pitching_moment(a,gamma,lengths,nodes,Np,v)
    # Find the quarter-chord pitching moment coefficient
    Cmqc = qc_pitching_moment(a,CL,Cmle)
    # Write VPM data into a csv file for plotting
    # write_csv(a,CL,Cmle,Cmqc)

    # Debugging
    np.set_printoptions(precision=5, suppress=True, linewidth=200, floatmode="fixed")
    # print("\n========== NODES ==========")
    # print(nodes)
    # print("\n========== LENGTHS ==========")
    # print(lengths)
    # print("\n========== CONTROL POINTS ==========")
    # print(control_points)
    # print("\n========== A-MATRIX ==========")
    # print(A)
    # print("\n========== B-MATRIX ==========")
    # print(B)
    # print("\n========== GAMMA ==========")
    # print(gamma)
    print("\n=========================================")
    print("           AERODYNAMIC RESULTS ")
    print("=========================================")
    print(f"V     = {V:12.5f}")
    print(f"CP    = {Cp:12.5f}")
    print(f"CL    = {CL:12.5f}")
    print(f"Cmle  = {Cmle:12.5f}")
    print(f"Cmqc  = {Cmqc:12.5f}")
    print("=========================================")
    print()

def airfoil_matrix(cp,lengths,nodes,Np,P,A):
    """ This is where the P-matrix, A-matrix, and normal vectors are made. """
    for i in range(Np): # Identifies the control point being influenced
        # Define the related control points
        cpx = cp[i,0]
        cpy = cp[i,1]
        # Find the nodal point differences
        dxi = nodes[i+1,0]-nodes[i,0]
        dyi = nodes[i+1,1]-nodes[i,1]
        # Find the correlated panel length being influenced
        li = lengths[i,0]
        # Find the normal vector at the control points
        nx = -dyi / li
        ny = dxi / li
        for j in range(Np): # Loops through each panel that influences a control point
            # Define the related node point
            x = nodes[j,0]
            y = nodes[j,1]
            # Find the nodal point differences
            dxj = nodes[j+1,0]-nodes[j,0]
            dyj = nodes[j+1,1]-nodes[j,1]
            # Find the cp difference with the node
            cpdx = cpx - x
            cpdy = cpy - y
            # Define the related panel length
            lj = lengths[j,0]
            # Find the panel coordinate locations
            eta,xi = panel_location(cpdx,cpdy,dxj,dyj,lj)
            # Find the Phi angle
            phi = np.arctan2(eta*lj,eta**2 + xi**2 - xi*lj)
            # Find the Psi
            psi = 0.5 * np.log((xi**2 + eta**2)/((xi-lj)**2 + eta**2))
            # Calculate intermediate terms
            a = (lj-xi)*phi + eta*psi
            b = xi*phi - eta*psi
            c = eta*phi - (lj-xi)*psi - lj
            d = -eta*phi - xi*psi + lj
            # Calculate P-matrix elements
            P00 = (dxj*a - dyj*c) / (2*pi*lj**2)
            P01 = (dxj*b - dyj*d) / (2*pi*lj**2)
            P10 = (dyj*a + dxj*c) / (2*pi*lj**2)
            P11 = (dyj*b + dxj*d) / (2*pi*lj**2)
            # Store the P-matrix coefficients
            P[0,0] = P00
            P[0,1] = P01
            P[1,0] = P10
            P[1,1] = P11
            # Add panel j's contributions to the A-matrix
            A[i,j] += nx*P[0,0] + ny*P[1,0]
            A[i,j+1] += nx*P[0,1] + ny*P[1,1]
    # Apply Kutta condition
    A[Np, 0] = 1.0
    A[Np, Np] = 1.0
    return A

def bmatrix(a,B,lengths,Np,nodes,v):
    """ This is where the B-matrix is built. The matrix shares how much normal 
        velocity the vortex panels must produce to prevent air from flowing 
        through the airfoil.
    """
    # Iterate through each node
    for i in range(Np):
        # Define the related x and y node values
        x1 = nodes[i,0]
        x2 = nodes[i+1,0]
        y1 = nodes[i,1]
        y2 = nodes[i+1,1]
        # Define the related panel length
        l = lengths[i,0]
        # Find the B-matrix
        B[i,0] = v * (((y2-y1)*np.cos(a)) - ((x2-x1)*np.sin(a)))/l
    # Define the last element to be zero
    B[Np] = 0.0
    return B

def get_control_points(control_points,nodes,Np):
    """ This is where the control points are calculated and stored. """
    for i in range(Np):
        control_points[i,0] = 0.5*(nodes[i,0]+nodes[i+1,0])
        control_points[i,1] = 0.5*(nodes[i,1]+nodes[i+1,1])
    return control_points

def le_pitching_moment(a,gamma,lengths,nodes,Np,v):
    """ This is where the section LE Pitching Moment Coefficient if found. """
    Cmle = 0.0
    for i in range(Np):
        # Define some convenient variables
        x1 = nodes[i,0]
        x2 = nodes[i+1,0]
        y1 = nodes[i,1]
        y2 = nodes[i+1,1]
        g1 = gamma[i,0]
        g2 = gamma[i+1,0]
        # Fine the pitching moment about the LE
        Cmle += lengths[i,0] * ((((2*x1*g1 + x1*g2 + x2*g1 + 2*x2*g2)/v)*np.cos(a)) \
                                + (((2*y1*g1 + y1*g2 + y2*g1 * 2*y2*g2)/v)*np.sin(a)))
    Cmle = -Cmle/3
    return Cmle

def lift(gamma,lengths,Np,v):
    """ This is where the lift coefficient is found. """
    CL = 0.0
    for i in range(Np):
        CL += lengths[i,0] * ((gamma[i,0]+gamma[i+1,0]) / v)
    return CL

def load_geometry_data(g):
    """ This imports the airfoil node coordinates from the geometry .txt file. """
    # Define the input path
    script_dir = Path(__file__).parent
    input_path = script_dir / g
    # Read geometry into an Nx2 array
    with open(input_path, 'r') as file:
        nodes = np.loadtxt(file)
    return nodes

def load_json_data():
    """ This imports all content from the .txt file. """
    # Define the input path
    script_dir = Path(__file__).parent
    input_path = script_dir/"input.json"
    # Open and load JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)
    # Pull JSON content out
    a = np.deg2rad(data["alpha[deg]"])
    v = data["freestream_velocity"]
    g = data["geometry"]
    return a, g, v

def nodes_and_panels(nodes):
    """ This finds the number of nodes and panels. """
    # Find the total number of nodes
    N = len(nodes)
    # Find the total number of panels
    Np = N - 1
    return N, Np

def nodal_vortex_strength(A,B,gamma):
    """ This is where the vortex strengths are found across the sheet at a node. """
    # Find the vortex strengths
    gamma = np.linalg.solve(A,B)
    return gamma

def panel_influence(i,lengths,nodes,xp,yp):
    """ This is where the P-matrix is made again but used for finding the
        induced velocity at a point influenced by the panels.
    """
    # Define the nodal points
    x = nodes[i,0]
    y = nodes[i,1]
    # Define the nodal point differences
    dx = nodes[i+1,0] - nodes[i,0]
    dy = nodes[i+1,1] - nodes[i,1]
    # Define the relative panel length
    l = lengths[i,0]
    # Define the control point and nodal point difference
    cpdx = xp - x
    cpdy = yp - y
    # Find airfoil coordinates of the panel at the control points
    eta, xi = panel_location(cpdx, cpdy, dx, dy, l)
    # Find phi and psi
    phi = np.arctan2(eta*l, eta**2 + xi**2 - xi*l)
    psi = 0.5*np.log((xi**2 + eta**2) / ((xi-l)**2 + eta**2))
    # Define some convienent elements for computing
    a = (l-xi)*phi + eta*psi
    b = xi*phi - eta*psi
    c = eta*phi - (l-xi)*psi - l
    d = -eta*phi - xi*psi + l
    # Preallocate matrix
    P = np.zeros((2,2))
    # Define the P-matrix elements
    P[0,0] = (dx*a - dy*c)/(2*pi*l**2)
    P[0,1] = (dx*b - dy*d)/(2*pi*l**2)
    P[1,0] = (dy*a + dx*c)/(2*pi*l**2)
    P[1,1] = (dy*b + dx*d)/(2*pi*l**2)
    return P

def panel_length(lengths,nodes,Np):
    """ This is where the length of the panels are calculated. """
    # Calculate the panel lengths
    for i in range(Np):
        dx = nodes[i+1,0]-nodes[i,0]
        dy = nodes[i+1,1]-nodes[i,1]
        lengths[i,0] = np.sqrt(dx**2 + dy**2)
    return lengths

def panel_location(cpdx,cpdy,dx,dy,l):
    """ This is to find the panel location in xi and eta terms of a control point. """
    # Find xi (parallel to panel)
    xi = ((dx*cpdx)+(dy*cpdy)) / l
    # Find eta (perpendicular to panel)
    eta = ((-dy*cpdx)+(dx*cpdy)) / l
    return eta, xi

def preallocate(N,Np):
    """ This creates pre-allocated variables. """
    # Panel lengths (Nx1 array)
    lengths = np.zeros((Np,1))
    # Control Points (Nx2 array)
    control_points = np.zeros((Np,2))
    # A-matrix (NxN array) - 
    A = np.zeros((N,N))
    # P-matrix (2x2 array) - velocity induced at the arbitrary point by a panel
    P = np.zeros((2,2))
    # B-matrix (Nx1 array)
    B = np.zeros((N,1))
    # Vortex strength
    gamma = np.zeros((N,1))
    return A,B,control_points,gamma,lengths,P

def pressure(V,v):
    """ This is where the pressure coefficient is found. """
    return 1 - (V**2 / v**2)

def qc_pitching_moment(a,CL,Cmle):
    """ This is where the quarter-chord pitching moment coefficient is found. """
    # Find the sectional normal coefficient
    CN = CL * np.cos(a)
    # Find the quarter-chord pitching moment coefficient
    Cmqc = Cmle + 0.25*CN
    return Cmqc

def velocity(a,cp,v,gamma,nodes,lengths,Np):
    """ This is where the velocity is found at a point. """
    # Find the freestream velocities
    vx = v * np.cos(a)
    vy = v * np.sin(a)
    # Find the induced velocities being influenced by the panels
    for i in range(Np):
        P = panel_influence(i,lengths,nodes,cp[i,0],cp[i,1])
        vx += P[0,0]*gamma[i,0] + P[0,1]*gamma[i+1,0]
        vy += P[1,0]*gamma[i,0] + P[1,1]*gamma[i+1,0]
    # Find the total velocity
    V = np.sqrt(vx**2 + vy**2)
    return V

# def write_csv(a,CL,Cmle,Cmqc):
#     """ This function writes the output to a csv file for plotting. """
#     # Define the headers
#     headers = ["alpha (deg)","CL_tilde","Cmle_tilde","Cmc/4_tilde"]
#     # Put data into lested list
#     vpm_data = []
#     for i in range():

#     # Write to the file - always specify newline="" and encoding=""
#     with open("vpm_results.csv", "w", newline="", encoding="utf-8") as file:


if __name__ == "__main__":
    # Constants
    pi = np.pi
    main()