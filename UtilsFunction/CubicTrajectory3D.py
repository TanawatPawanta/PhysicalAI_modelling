from pydrake.all import PiecewisePolynomial, LeafSystem, BasicVector
import numpy as np

class CubicTrajectory3D(LeafSystem):
    def __init__(self):
        """
        Initializes the 3D trajectory generator using cubic Hermite splines.

        Args:
            times (list): Time points for the waypoints.
            positions_xyz (np.ndarray): 3xN array of positions for each axis (x, y, z).
            velocities_xyz (np.ndarray): 3xN array of velocities for each axis (x, y, z).
        """
        super().__init__()
        #  Variables
        self.z_max = 0.04
        self.time_step = 0.01  # [s]
        self.state = 0
        self.start_time = 0.0
        self.curr_positions = np.array([
                    [0.0],
                    [0.0],
                    [0.0],
                ])
        self.curr_velocities = np.array([
                    [0.0],
                    [0.0],
                    [0.0],
                ])                                                                                                                                                                                                                                                                                      
        self.traj_x = None
        self.traj_y = None
        self.traj_z = None

        self.DeclarePeriodicUnrestrictedUpdateEvent(period_sec=self.time_step                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ,
                                                    offset_sec=0.0,
                                                    update=self.Update)

        # Input
        self.input_port0 = self.DeclareVectorInputPort("foot_placements", BasicVector(6))
        self.input_port1 = self.DeclareVectorInputPort("step_time", BasicVector(1))
        self.input_port2 = self.DeclareVectorInputPort("gen_command",BasicVector(1))
        # Outputs: Position, Velocity, and Acceleration (3D vectors)
        self.DeclareVectorOutputPort("Position", BasicVector(3), self.CalcPosition)
        self.DeclareVectorOutputPort("Velocity", BasicVector(3), self.CalcVelocity)
        # self.DeclareVectorOutputPort("velocity", BasicVector(3), self.CalcVelocity)
        # self.DeclareVectorOutputPort("acceleration", BasicVector(3), self.CalcAcceleration)

    def Update(self,context,state):
        gen_command = self.get_input_port(2).Eval(context)[0] # 0:idle 1:start 2:reset
        system_time = context.get_time()
        # print("state: ",self.state)
        match self.state:
            case 0: # Idle
                if gen_command == 1: # State transition
                    # Initiate trajectory 
                    self.state = 1
                    self.start_time = context.get_time()
                    self.traj_x, self.traj_y, self.traj_z = self.InitTrajectory3D(context=context)
                    relative_time = system_time - self.start_time
                    self.curr_positions = np.array([
                        self.traj_x.value(relative_time)[0],
                        self.traj_y.value(relative_time)[0],
                        self.traj_z.value(relative_time)[0],
                    ])
            case 1: # Evaluate trajectory
                relative_time = system_time - self.start_time
                self.curr_positions = np.array([
                    self.traj_x.value(relative_time)[0],
                    self.traj_y.value(relative_time)[0],
                    self.traj_z.value(relative_time)[0],
                ])   
                self.curr_velocities = np.array([
                    self.traj_x.EvalDerivative(relative_time, 1)[0],
                    self.traj_y.EvalDerivative(relative_time, 1)[0],
                    self.traj_z.EvalDerivative(relative_time, 1)[0],
                ])
                if gen_command == 2: # State transition  
                    self.state = 0         
    
    def InitTrajectory3D(self, context):
        foot_placements = self.get_input_port(0).Eval(context)
        step_time = self.get_input_port(1).Eval(context)[0]
        time_xy = np.array([0.0, step_time])
        time_z = np.array([0.0, step_time*0.5, step_time])
        
        px = np.array([[foot_placements[0],foot_placements[3]]])
        py = np.array([[foot_placements[1],foot_placements[4]]])
        pz = np.array([[foot_placements[2],self.z_max,foot_placements[5]]])
        vx = np.array([[0.0, 0.0]])
        vy = np.array([[0.0, 0.0]])
        vz = np.array([[0.0, 0.0, 0.0]])
        trajectory_x = PiecewisePolynomial.CubicHermite(time_xy, px, vx)
        trajectory_y = PiecewisePolynomial.CubicHermite(time_xy, py.reshape(1, -1), vy)
        trajectory_z = PiecewisePolynomial.CubicHermite(time_z, pz.reshape(1, -1), vz)
        return trajectory_x,trajectory_y,trajectory_z
    
    def CalcPosition(self, context, output):
        output.SetFromVector(self.curr_positions)

    def CalcVelocity(self, context, output):
        output.SetFromVector(self.curr_velocities)

    # def CalcAcceleration(self, context, output):
    #     time = self.get_input_port(0).Eval(context)[0]
    #     acceleration = np.array([
    #         self.trajectory_x.derivative(2).value(time)[0],
    #         self.trajectory_y.derivative(2).value(time)[0],
    #         self.trajectory_z.derivative(2).value(time)[0],
    #     ])
    #     output.SetFromVector(acceleration)
