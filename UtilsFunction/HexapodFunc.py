from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
import numpy as np

def CalcJointPositionFromRobotHeigh(plant,desired_base_heigh,desired_leg_lenght,):
    IK_solver = InverseKinematics(plant)
    end_effector_names = [
                        "front_left_Endeffector",  "front_right_Endeffector",
                        "middle_left_Endeffector", "middle_right_Endeffector",
                        "rear_left_Endeffector",   "rear_right_Endeffector"]
    hip_offset_angle = [ 
                        -np.pi/3.0,  -2.0*np.pi/3.0, 
                        0, -np.pi, 
                        -5.0*np.pi/3.0, -4.0*np.pi/3.0]
    base_frame = plant.GetFrameByName("BaseLink")

    # Add position constraints
    for i in range(6):
        end_effector_frame = plant.GetFrameByName(end_effector_names[i])
        eff_P_base = RotationMatrix.MakeZRotation(hip_offset_angle[i]) @ np.array([0.0,desired_leg_lenght,-1.0*desired_base_heigh])
        IK_solver.AddPositionConstraint(
            frameB=end_effector_frame,
            p_BQ=np.zeros(3),     # point in end-effector frame
            frameA=base_frame,
            p_AQ_lower=eff_P_base-0.001,
            p_AQ_upper=eff_P_base+0.001
        )
    q_variables = IK_solver.q()
    initial_guess = np.ones(plant.num_positions())*0.1
    result = Solve(IK_solver.prog(),initial_guess)
    q = None
    if result.is_success():
        q_solution = result.GetSolution(q_variables)
        q = q_solution
    else:
        print("Failed to find IK solution!")
    return q