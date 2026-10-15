import numpy as np # does math stuff    
import json # interfaces with .json input file 
import matplotlib.pyplot as plt # does plotting stuff

def load_json(json_filename):
    # extract json dictionary info
    json_string = open(json_filename).read() # read it as txt
    input_vals = json.loads(json_string) # load it as a json dictionary using the json package
    # now the info is easy to extract! 
    geometry = input_vals['geometry']  
    alpha_rad = np.deg2rad(input_vals["alpha[deg]"])
    vel_inf = input_vals["freestream_velocity"]
    return geometry, alpha_rad, vel_inf

def get_nodes_from_geom(geometry): 
    # this get's the nodes from the geometry txt file 
    with open(geometry, "r") as f:
        info = []
        for line in f:
            info.append([float(coordinate) for coordinate in line.split()])
    # now make the info list a numpy array 
    nodes = np.array(info)
    return nodes

def calc_panel_lengths(N_panels, etcetera):
    lengths = np.zeros((N_panels,1)) # array of panel lengths
    ## calculate the panel lengths ##

def calc_control_point_locations(N_panels, etcetera):
    control_point_locations = np.zeros((N_panels-1,2)) # array of control points 
    ## calculate the control point locations ##

if __name__ == "__main__": # Instructions after this line are what the code is actually following. 
    json_filename = "input.json" 
    geometry, alpha_rad, vel_inf = load_json(json_filename)
    print("\ngeom:\n", geometry)
    print("\nalpha [deg]:\n", np.rad2deg(alpha_rad))
    print("\nFreestream velocity:\n", vel_inf)
    nodes = get_nodes_from_geom(geometry)
    print("\nnodes:\n", nodes)
    N_nodes = len(nodes)
    print("\nN (nodes):\n",N_nodes)
    N_panels = N_nodes - 1 
    print("\nN (panels):\n",N_panels)

