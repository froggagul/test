import time
from typing import Union

import numpy as np
import pybullet as pb
import pybullet_data


class Branch:

    def __init__(
        self,
        start_point: np.ndarray,
        orientation: np.ndarray,
        thickness: float,
        length: float,
    ) -> None:
        self.start_point = start_point
        self.thickness = thickness
        self.length = length
        self.orientation = orientation

        rgba_color = [1.0, 1.0, 1.0, 1.0]
        self._collision_shape = pb.createCollisionShape(
            shapeType=pb.GEOM_CYLINDER, radius=thickness, height=length)
        self._visual_shape = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER,
                                                  radius=thickness/2,
                                                  length=length,
                                                  rgbaColor=rgba_color)

        self.end_point = start_point + pb.rotateVector(
            orientation, np.array([0, 0, length]))

        position = (start_point + self.end_point) / 2

        self.id = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self._collision_shape,
            baseVisualShapeIndex=self._visual_shape,
            basePosition=position,
            baseOrientation=orientation)

        self.child_branches = []

    def add_child_branch(
        self,
        fraction: float,
        orientation: np.ndarray,
        thickness: float,
        length: float,
    ):
        start_point = self.start_point + (self.end_point -
                                          self.start_point) * fraction
        branch = Branch(
            start_point,
            orientation,
            thickness,
            length,
        )
        self.child_branches.append(branch)

    def get_ids(self):
        ids = [self.id]
        for branch in self.child_branches:
            ids += branch.get_ids()
        return ids


class Hanger:

    def __init__(self, num_branches: int = 1,
                 base_start_point: np.ndarray = np.array([0, 0, 0]),
                 base_thickness: float = 0.02,
                 base_height: float =0.5,
                 ) -> None:

        self.base_branch = Branch(
            base_start_point,
            np.array([0, 0, 0, 1]),
            base_thickness,
            base_height,
        )

        for i in range(num_branches):
            fraction = 0.5 + 0.5 * np.random.rand()
            orientation = pb.getQuaternionFromEuler(
                [np.pi * 0.2, 0, np.random.rand() * 2 * np.pi])
            length = 0.15
            thickness = 0.01
            self.base_branch.add_child_branch(
                fraction,
                orientation,
                thickness,
                length,
            )


if __name__ == "__main__":
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF("plane.urdf")
    pb.setGravity(0, 0, -9.8)

    hanger = Hanger(num_branches=3)

    pb.setTimeStep(1 / 60)
    for _ in range(60 * 5):
        pb.stepSimulation()
        time.sleep(1 / 60)
    pb.disconnect()
