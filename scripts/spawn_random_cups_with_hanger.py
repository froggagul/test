import os
import random
import time

import numpy as np
import pybullet as pb
import pybullet_data

from hanger import Hanger

if __name__ == "__main__":
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF("plane.urdf")
    pb.setGravity(0, 0, -9.8)

    hanger = Hanger(num_branches=3)

    for i in range(5):
        position = np.random.rand(3) * np.array([0.5, 0.5, 1.0]) + np.array(
            [-0.25, -0.25, 1])
        mass = 0.1
        scale = 0.3

        assets_path = os.path.join(os.path.dirname(__file__), "../assets")
        obj_files = [f for f in os.listdir(assets_path) if f.endswith('.obj')]

        if obj_files:
            obj_file = random.choice(obj_files)
            obj_path = os.path.join(assets_path, obj_file)
            visual_shape_id = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                                   fileName=obj_path,
                                                   meshScale=[scale] * 3)
            collision_shape_id = pb.createCollisionShape(
                shapeType=pb.GEOM_MESH,
                fileName=obj_path,
                meshScale=[scale] * 3)
            pb.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=collision_shape_id,
                               baseVisualShapeIndex=visual_shape_id,
                               basePosition=position)

    pb.setTimeStep(1 / 240)
    for _ in range(240 * 5):
        pb.stepSimulation()
        time.sleep(1 / 240)
    pb.disconnect()
