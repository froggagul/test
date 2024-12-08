import os
import random
import time

import pybullet as pb
import pybullet_data


if __name__ == "__main__":
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF("plane.urdf")
    pb.setGravity(0, 0, -9.8)

    assets_path = os.path.join(os.path.dirname(__file__), "../assets")
    obj_files = [f for f in os.listdir(assets_path) if f.endswith('.obj')]

    if obj_files:
        obj_file = random.choice(obj_files)
        obj_path = os.path.join(assets_path, obj_file)
        visual_shape_id = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName=obj_path)
        collision_shape_id = pb.createCollisionShape(shapeType=pb.GEOM_MESH, fileName=obj_path)
        pb.createMultiBody(baseMass=1, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=[0, 0, 1])

    pb.setTimeStep(1 / 60)
    for _ in range(60 * 5):
        pb.stepSimulation()
        time.sleep(1 / 60)
    pb.disconnect()