import os
import random
import time
import functools
import numpy as np
import pybullet as pb
import pybullet_data
import jax
from train_ddpm import MLP
from scripts.data_load import get_dataloader
import jax.numpy as jnp
import pickle

from scripts.hanger import Hanger
from train_ddpm import p_sample_loop

def get_pointcloud(uids, pointcloud_num):
    total_surface_points = []
    for uid in uids:
        aabb = pb.getAABB(uid)
        min_bound = np.array(aabb[0])
        max_bound = np.array(aabb[1])

        # Generate random ray origins and directions within the bounding box
        num_rays = 5000  # Number of rays
        origins = np.random.uniform(min_bound, max_bound, size=(num_rays, 3))
        directions = np.random.normal(size=(num_rays, 3))  # Random directions
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)  # Normalize directions

        results = pb.rayTestBatch(origins, origins + directions * 100)  # Scale to long rays
        surface_points = []

        # Collect hit points
        for result in results:
            hit_object_id, hit_position, _ = result[0], result[3], result[2]
            if hit_object_id in uids:  # Check if hitObjectUniqueId matches the mesh_id
                surface_points.append(hit_position)

        if len(surface_points) > 0:
            surface_points = np.array(surface_points)
            total_surface_points.append(surface_points)

    total_surface_points = np.concatenate(total_surface_points, axis=0)
    # random select pointcloud
    if pointcloud_num < total_surface_points.shape[0]:
        total_surface_points = total_surface_points[np.random.choice(total_surface_points.shape[0], pointcloud_num, replace=False)]
    return total_surface_points

def visualize_pointcloud(pointcloud):
    for point in pointcloud:
        pb.addUserDebugLine(point, point + np.array([0, 0, 0.01]), [1, 0, 0], 2)

def test_simulation(num_branches=1, pointcloud_num=256, inference=None):
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF("plane.urdf")
    pb.setGravity(0, 0, -9.8)

    hanger = Hanger(num_branches=num_branches)

    mug_uids = []
    branch_uids = hanger.base_branch.get_ids()
    branch_surface_points = get_pointcloud(branch_uids, pointcloud_num)
    mug_surface_points = {}

    for j in range(num_branches):
        scale = 0.3

        branch_endpoint = hanger.base_branch.child_branches[j].end_point
        position = branch_endpoint + np.array([-0.3 *scale, 0, 0])
        position[2] = 1 + j * 0.2
        rgba_color = np.random.rand(4)
        rgba_color[3] = 1

        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        obj_files = [f for f in os.listdir(assets_path) if f.endswith('.obj')]

        if obj_files:
            obj_file = random.choice(obj_files)
            obj_path = os.path.join(assets_path, obj_file)
            visual_shape_id = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                                   fileName=obj_path,
                                                   rgbaColor=rgba_color,
                                                   meshScale=[scale] * 3)
            collision_shape_id = pb.createCollisionShape(
                shapeType=pb.GEOM_MESH,
                fileName=obj_path,
                meshScale=[scale] * 3)
            mug_uid = pb.createMultiBody(baseMass=1,
                                         baseCollisionShapeIndex=collision_shape_id,
                                         baseVisualShapeIndex=visual_shape_id,
                                         basePosition=position)
            mug_uids.append(mug_uid)
            mug_surface_points[mug_uid] = get_pointcloud([mug_uid], pointcloud_num) - np.array(position).reshape(1, 3)

        mug_pose = inference(jnp.array(branch_surface_points).reshape(1, 256, 3), jnp.array(mug_surface_points[mug_uid]).reshape(1, 256, 3))
        pb.resetBasePositionAndOrientation(mug_uid, mug_pose[0, :3], mug_pose[0, 3:])
    breakpoint()
    # stabilize
    pb.setTimeStep(1 / 240)
    for _ in range(240 * 1):
        pb.stepSimulation()
        time.sleep(1 / 240)

    # find cup hanging on hanger by checking collision for each branch, remove mug with colliding floor
    collision_informations = pb.getContactPoints()
    collision_informations = list(set([(info[1], info[2]) if info[1] < info[2] else (info[2], info[1]) for info in collision_informations]))
    mug_collision_with_branch = []
    mug_collision_with_floor = []
    for collision_information in collision_informations:
        if collision_information[0] == 0 and collision_information[1] in mug_uids:
            mug_collision_with_floor.append(collision_information[1])
        if collision_information[0] in branch_uids and collision_information[1] in mug_uids:
            mug_collision_with_branch.append(collision_information[1])
    mug_uids = list(set(mug_collision_with_branch) - set(mug_collision_with_floor))
    pb.disconnect()

    if len(mug_uids) > 0:
        return True
    else:
        return False

if __name__ == "__main__":
    _, val_dataloader = get_dataloader("dataset", 1, 1)
    model = MLP()

    rng = jax.random.PRNGKey(0)
    batch = next(iter(val_dataloader))
    sample_t = jnp.array([0], dtype=jnp.int32)
    mug_pose = batch['mug_poses'][0:1]
    branch_pcs = batch['branch_pcs'][0:1]
    mug_pcs = batch['mug_pcs'][0:1]

    params = model.init(rng, mug_pose, sample_t, branch_pcs, mug_pcs)
    ckpt_path = "checkpoint/save_dict.pkl"
    with open(ckpt_path, "rb") as f:
        save_dict = pickle.load(f)
    params = save_dict["params"]

    # samples = p_sample_loop(model, params, rng, (1, 7), branch_pcs, mug_pcs)
    inference = functools.partial(p_sample_loop, model, params, rng, (1, 7))
    print(inference(branch_pcs, mug_pcs).shape)
    test_simulation(inference=inference)