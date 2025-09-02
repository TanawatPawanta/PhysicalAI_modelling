from scipy.spatial.transform import Rotation as R
from IPython.display import SVG, display
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import MultibodyForces
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.primitives import LogVectorOutput
from pydrake.geometry import Meshcat, Sphere, Rgba, Box
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pydot

def quaternion_to_euler(Q)->list:
    w = Q[0]
    x = Q[1]
    y = Q[2]
    z = Q[3]
    r = R.from_quat([x, y, z, w])  # Note: SciPy uses [x, y, z, w] order
    out = r.as_euler('xyz', degrees=True)
    return [float(out[0]),float(out[1]),float(out[2])] # Returns roll, pitch, yaw

def Jleg_position(J_hip_y_q=0.0, J_hip_r_q=0.0, J_hip_p_q=0.0, J_knee_q=0.0, J_ankle_p_q=0.0, J_ankle_r_q=0.0):
    """
    Args:
        joint angles (float, optional): joint angles in deg. Defaults to 0.0.

    Returns:
        np.array: joint angles in rad
    """
    return np.array([[J_hip_y_q, J_hip_r_q, J_hip_p_q, J_knee_q, J_ankle_p_q, J_ankle_r_q]]) * np.pi/180.0

def Jarm_position(JL_shoulder_p_q=0.0, JL_shoulder_r_q=0.0, JL_elbow_q=0.0):
    return np.array([[JL_shoulder_p_q, JL_shoulder_r_q, JL_elbow_q]]) * np.pi/180.0

def Jhead_position(J_neck_q=0.0, J_cam_q=0.0):
    return np.array([[J_neck_q, J_cam_q]]) * np.pi/180.0

def show_diagram(diagram, show=True):
    if show:
        display(
        SVG(
            pydot.graph_from_dot_data(
                diagram.GetGraphvizString(max_depth=1))[0].create_svg()))
    else:
        pass

def ListJointsName(plant,model_instance_index):
    joint_ind = plant.GetJointIndices(model_instance_index[0]) # all joint index in plant
    print("GetJointIndices: ",joint_ind)
    for i in joint_ind:
        joint = plant.get_joint(i)  # get constant joint reference(read only) from plant
        print(f"Joint {i}: {joint.name()}")

def ListActuatorName(plant,model_instance_index):
    joint_ind = plant.GetJointActuatorIndices(model_instance_index) # all actuate joint index in plant
    for i in joint_ind:
        actuator = plant.get_joint_actuator(i)  # get constant actuate joint reference(read only) from plant
        print(f"Actuator {i}: {actuator.name()}")

def ListPlantOutputPort(plant):
    num_ports = plant.num_output_ports()    # number of output port of Multibody plant
    for i in range(num_ports):
        output_port = plant.get_output_port(i)
        print(f"Output Port {i}: {output_port.get_name()}, Type: {output_port.get_data_type()}")    

def ListStatesNames(plant):
    states_names = np.array([plant.GetStateNames()]).T
    print("Num state: ",states_names.size," states")
    print("Num position: ",plant.num_positions())
    print("Num velocity: ",plant.num_velocities())
    print(states_names)

def ListFrameNames(plant,model_instance_index):
    """
    Returns a list of all frame names in the given MultibodyPlant.

    Args:
        plant: An instance of pydrake.multibody.plant.MultibodyPlant.

    Returns:
        A list of strings representing the names of all frames in the plant.
    """
    frameID = plant.GetFrameIndices(model_instance_index[0])
    print("Robot frames name")
    for i in frameID:
        print(i,"->",plant.get_frame(i).name())

def ListBodyNames(plant, model_instance_index):
    for i in plant.GetBodyIndices(model_instance=model_instance_index[0]):
        body = plant.get_body(i)
        print(f"{i}: {body.name()}")

def RigidTransformToMatrix(rigid_transform: RigidTransform) -> np.ndarray:
    """
    Converts a Drake RigidTransform to a 4x4 NumPy homogeneous transformation matrix.

    Args:
        rigid_transform (RigidTransform): The RigidTransform to convert.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the homogeneous transformation matrix.
    """
    # Access the rotation matrix and convert to NumPy array
    rotation_matrix_np = rigid_transform.rotation().matrix()  # Shape: (3, 3)

    # Access the translation vector as a NumPy array
    translation_vector_np = rigid_transform.translation()     # Shape: (3,)

    # Initialize a 4x4 identity matrix
    homogeneous_matrix = np.eye(4)

    # Assign the rotation matrix to the top-left 3x3 block
    homogeneous_matrix[:3, :3] = rotation_matrix_np

    # Assign the translation vector to the top-right 3x1 block
    homogeneous_matrix[:3, 3] = translation_vector_np

    return homogeneous_matrix

def DisplayGait(x_data, y_data, rectangle_centers, rectangle_size, description):
    """
    Plot a trajectory and rectangles in the same plot with equal resolution and a description below the plot.

    Parameters:
        x_data (list or array): X-axis data for the trajectory.
        y_data (list or array): Y-axis data for the trajectory.
        rectangle_centers (list of tuples): List of (x, y) coordinates for rectangle centers.
        rectangle_size (tuple): Size of rectangles (width, height).
        description (str): Description to display below the plot.

    Returns:een" if center[1] < 0 else "red"
        rectangle = patches.Rectangle(
            (rect_x+0.015, rect_y),
            rectangle_size[0],
            rectangle_size[1],
            edgecolor=edge_color,
            facecolor=e
        None: Displays the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    # Add rectangles
    for center in rectangle_centers:
        rect_x = center[0] - rectangle_size[0] / 2  # Bottom-left corner x
        rect_y = center[1] - rectangle_size[1] / 2  # Bottom-left corner y
        edge_color = "green" if center[1] < 0 else "red"
        rectangle = patches.Rectangle(
            (rect_x+0.015, rect_y),
            rectangle_size[0],
            rectangle_size[1],
            edgecolor=edge_color,
            facecolor=edge_color,
            alpha=0.3, 
            linewidth=2,
        )
        ax.add_patch(rectangle)
    # Plot the trajectory
    ax.scatter(x_data, y_data, label="CoM Trajectory", color="blue",s=0.1, linewidth=1)

    # Set equal scaling for x and y axes
    ax.set_aspect("equal")

    # Set equal resolution by matching axis limits
    x_min, x_max = min(x_data) - 0.1, max(x_data) + 0.1
    y_min, y_max = min(y_data) - 0.1, max(y_data) + 0.1
    max_range = max(x_max - x_min, y_max - y_min)

    # Set tick steps to 0.1
    x_ticks = np.arange(x_min, x_max, 0.1)
    y_ticks = np.arange(-0.1, 0.15, 0.1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Set plot labels and legend
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('LIP CoM Trajectory : '+description)
    ax.legend()
    ax.grid(True)

    # Display the plot
    plt.show()

def DisplayGait2(x_data, y_data,x_data2, y_data2, rectangle_centers, rectangle_size,des):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    # Add rectangles
    for center in rectangle_centers:
        rect_x = center[0] - rectangle_size[0] / 2  # Bottom-left corner x
        rect_y = center[1] - rectangle_size[1] / 2  # Bottom-left corner y
        edge_color = "green" if center[1] < 0 else "red"
        rectangle = patches.Rectangle(
            (rect_x+0.015, rect_y),
            rectangle_size[0],
            rectangle_size[1],
            edgecolor=edge_color,
            facecolor=edge_color,
            alpha=0.3, 
            linewidth=2,
        )
        ax.add_patch(rectangle)
    # Plot the trajectory
    ax.scatter(x_data, y_data, label="Desire CoM Trajectory", color="blue",s=0.1, linewidth=1)
    ax.plot(x_data2, y_data2, label=f"Actual {des} Trajectory", color="black", linewidth=1)

    # Set equal scaling for x and y axes
    ax.set_aspect("equal")

    # Set equal resolution by matching axis limits
    x_min, x_max = min(x_data) - 0.1, max(x_data) + 0.1
    y_min, y_max = min(y_data) - 0.1, max(y_data) + 0.1
    max_range = max(x_max - x_min, y_max - y_min)

    # Set tick steps to 0.1
    x_ticks = np.arange(-0.05, x_max, 0.1)
    y_ticks = np.arange(-0.1, 0.15, 0.1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Set plot labels and legend
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('CoM Trajectory Tracking Result')
    ax.legend()
    ax.grid(True)

    # Display the plot
    plt.show()

def CreateLoggers(builder, system, port_num:list[int], name:list[str]):
    out = []
    n = 0
    for i in port_num:
        logger = LogVectorOutput(system.get_output_port(i), builder)
        logger.set_name(name[n])
        n+=1
        out.append(logger)
    return out

def DefinePlant(model_name:str,is_fixed:bool,fixed_frame:str)->MultibodyPlant:
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    urdf = model_name+".urdf"
    dir = "/home/tanawatp/Documents/Hexapod/PhysicalAI_modelling/Models" # ROG path
    robot_urdf_path = os.path.join(dir,urdf)
    robot_instant_index = parser.AddModels(robot_urdf_path)
    if is_fixed:
        robot_fixed_frame = plant.GetFrameByName(fixed_frame)
        init_pos = [0.0, 0.0, 0.0] 
        init_orien = np.asarray([0, 0, 0])
        X_WRobot = RigidTransform(
        RollPitchYaw(init_orien * np.pi / 180), p=init_pos)
        plant.WeldFrames(plant.world_frame(),robot_fixed_frame, X_WRobot)
    plant.Finalize()
    return plant, robot_instant_index

def IsInsideRectangle(point, rect):
    """
    Check if a point is inside a rectangle.

    Parameters:
        point: np.array([x, y])
        rect: np.array([x_min, y_min, x_max, y_max])

    Returns:
        True if point is inside or on the edge of the rectangle.
    """
    x, y = point
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max

def PlotRectangleAndPoint(rect, point,resolution = 0.01):
    """
    Plot the rectangle and the point.

    Parameters:
        rect: np.array([x_min, y_min, x_max, y_max])
        point: np.array([x, y])
    """
    x_min, y_min, x_max, y_max = rect
    px, py = point

    fig, ax = plt.subplots()

    # Create rectangle patch
    width = x_max - x_min
    height = y_max - y_min
    rectangle = patches.Rectangle((x_min, y_min), width, height,
                                  linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rectangle)

    # Plot point
    color = 'go' if IsInsideRectangle(point, rect) else 'ro'
    ax.plot(px, py, color)

    # Adjust plot limits
    ax.set_xlim(min(x_min, px) - resolution, max(x_max, px) + resolution)
    ax.set_ylim(min(y_min, py) - resolution, max(y_max, py) + resolution)
    ax.set_aspect('equal')
    ax.grid(True)

    plt.title("Point Inside Rectangle" if IsInsideRectangle(point, rect) else "Point Outside Rectangle")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()     

def DisplayPointTrajectory(meshcat:Meshcat,trajectory_x,trajectory_y,trajectory_z):
    for i in range(trajectory_x.shape[0]):
        meshcat.SetObject(f"waypoints/pt_{i}", Sphere(0.001), rgba=Rgba(0, 0, 1, 1))
        p = [trajectory_x[i],trajectory_y[i],trajectory_z[i]]
        meshcat.SetTransform(f"waypoints/pt_{i}", RigidTransform(p))

def DisplayFootPlacementArea(meshcat:Meshcat,foot_placement_position,foot_size):
    foot_area = Box(foot_size[0], foot_size[1], 0.001) 
    ind = 0
    for p in foot_placement_position:
        if p[1] > 0:
            rgb = Rgba(1, 0, 0, 0.5)
        else:
            rgb = Rgba(0, 1, 0, 0.5)
        meshcat.SetObject(f"footstep_area_{ind}", foot_area, rgba=rgb)  # translucent green
        pose = RigidTransform([p[0]+0.015, p[1], 0.001/2])
        meshcat.SetTransform(f"footstep_area_{ind}", pose)
        ind += 1

def PlotjointTrajectories(time_points, joint_data, joint_names=None, 
                                  title="Robot Joint Trajectories", 
                                  legend_labels=None,
                                  colors=None,
                                  line_styles=None,
                                  fig_size=(15, 12),
                                  save_path=None,
                                  show_plot=True):
    """
    Plot multiple robot joint trajectories for comparison.
    
    Parameters:
    -----------
    time_points : array-like
        Time points for the trajectory data
    joint_data : list of arrays
        List containing arrays of joint position data. Each array should have shape
        (n_time_points, n_joints) where n_joints is the number of joints (e.g., 6)
    joint_names : list, optional
        Names of the joints (default: ["Joint 1", "Joint 2", ..., "Joint n"])
    title : str, optional
        Title for the entire plot
    legend_labels : list, optional
        Labels for the legend to identify different trajectories
        (default: ["Trajectory 1", "Trajectory 2", ...])
    colors : list, optional
        Colors for each trajectory (default: automatically chosen)
    line_styles : list, optional
        Line styles for each trajectory (default: ['-', '--', '-.', ':', '-+'])
    fig_size : tuple, optional
        Figure size as (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    show_plot : bool, optional
        Whether to display the plot (default: True)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    # Determine the number of joints from the first trajectory data
    n_joints = joint_data[0].shape[1]
    
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(n_joints)]
    
    if legend_labels is None:
        legend_labels = [f"Trajectory {i+1}" for i in range(len(joint_data))]
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':']
        # Extend if more trajectories
        if len(joint_data) > 5:
            line_styles = line_styles * ((len(joint_data) // 5) + 1)
    
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        # Extend if more trajectories
        if len(joint_data) > 5:
            colors = colors * ((len(joint_data) // 5) + 1)
    
    # Calculate grid layout based on number of joints
    if n_joints <= 3:
        n_rows, n_cols = 1, n_joints
    elif n_joints <= 6:
        n_rows, n_cols = 2, 3
    elif n_joints <= 9:
        n_rows, n_cols = 3, 3
    else:
        n_rows = (n_joints + 3) // 4  # Ceiling division
        n_cols = 4
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(n_rows, n_cols, figure=fig)
    axes = []
    
    # Create subplot for each joint
    for i in range(n_joints):
        row = i // n_cols
        col = i % n_cols
        axes.append(fig.add_subplot(gs[row, col]))
    
    # Plot each joint's trajectory
    for joint_idx in range(n_joints):
        ax = axes[joint_idx]
        
        # Plot each trajectory for this joint
        for traj_idx, trajectory in enumerate(joint_data):
            ax.plot(time_points, trajectory[:, joint_idx], 
                   label=legend_labels[traj_idx],
                   linestyle=line_styles[traj_idx % len(line_styles)],
                   color=colors[traj_idx % len(colors)],
                   linewidth=2)
        
        ax.set_title(joint_names[joint_idx])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Joint Position (rad)')
        ax.grid(True)
        
        # Only add legend to the first subplot to avoid redundancy
        if joint_idx == 0:
            ax.legend(loc='best')
    
    # Add a main title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Adjust for the suptitle
    plt.subplots_adjust(top=0.9)
    
    # Save figure if specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if specified
    if show_plot:
        plt.show()
    return fig

from pydrake.multibody.tree import JointActuatorIndex
def PrintJointTorque(plant,tau):
    print("Joints torque")
    for i in range(plant.num_actuators()):                     # ← size nᵤ  :contentReference[oaicite:0]{index=0}
        act = plant.get_joint_actuator(JointActuatorIndex(i))  # singular accessor
        j   = act.joint()                                      # the joint it drives
        i0  = j.velocity_start()                               # first v‑index for that joint  :contentReference[oaicite:1]{index=1}
        nv  = j.num_velocities()

        print(f"{act.name():>20}: τ = {tau[i0:i0+nv].flatten()}") 