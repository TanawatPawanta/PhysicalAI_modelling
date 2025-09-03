from pydrake.all import LeafSystem, BasicVector
from enum import Enum


class DummyParameters2(LeafSystem):
    def __init__(self,T0_swing,T_swing,step_lenght):
        super().__init__()  # Don't forget to initialize the base class.

        self.DeclareNumericParameter(BasicVector([0]))
        self.DeclareNumericParameter(BasicVector([0]))
        self.DeclareNumericParameter(BasicVector([0.0, 0.0, 0.0, step_lenght*2.0, 0.0,0.0]))
        self.DeclareNumericParameter(BasicVector([T_swing]))
        self.DeclareNumericParameter(BasicVector([0]))
        self.DeclareNumericParameter(BasicVector([1]))
        self.T0_swing = T0_swing

        self.DeclareVectorOutputPort(name="foot_placements",
                                     size=6,
                                     calc=self.Output_FP)
        self.DeclareVectorOutputPort(name="step_time",
                                     size=1,
                                     calc=self.Output_step_time)
        self.DeclareVectorOutputPort(name="gen_command",
                                     size=1,
                                     calc=self.Output_gen_command)
        

    def OutputL(self, context, output):
        output.SetFromVector(self.L_leg)

    def OutputR(self, context, output):
        output.SetFromVector(self.R_leg)

    def Output_FP(self, context, output):
        output.SetFromVector(context.get_numeric_parameter(2).get_value())

    def Output_step_time(self, context, output):
        output.SetFromVector(context.get_numeric_parameter(3).get_value())

    def Output_gen_command(self, context, output):
        time = context.get_time()
        if time >= self.T0_swing:
            output.SetFromVector(context.get_numeric_parameter(5).get_value())
        else:
            output.SetFromVector(context.get_numeric_parameter(4).get_value())

    

