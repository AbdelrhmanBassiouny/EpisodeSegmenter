import datetime
import json
import os
import shutil
import threading
import time
from abc import ABC, abstractmethod

import numpy as np
from tf.transformations import quaternion_from_matrix, euler_matrix
from typing_extensions import List, Tuple, Dict, Optional

from pycram.datastructures.enums import ObjectType
from pycram.datastructures.pose import Pose, Transform
from pycram.world_concepts.world_object import Object

try:
    from pycram.worlds.multiverse import Multiverse
except ImportError:
    Multiverse = None

from pycram.datastructures.world import World


class EpisodePlayer(threading.Thread, ABC):
    def __init__(self, time_between_frames: datetime.timedelta = datetime.timedelta(milliseconds=0)):
        super().__init__()
        self._ready = False
        self._pause: bool = False
        self.time_between_frames: datetime.timedelta = time_between_frames

    @property
    def ready(self):
        return self._ready

    @abstractmethod
    def run(self):
        """
        The run method that is called when the thread is started. This should start the episode player thread.
        """
        pass

    def pause(self):
        """
        Pause the episode player frame processing.
        """
        self._pause: bool = True

    def resume(self):
        """
        Resume the episode player frame processing.
        """
        self._pause: bool = False

    def _wait_if_paused(self):
        """
        Wait if the episode player is paused.
        """
        while self._pause:
            time.sleep(0.1)

    def _wait_to_maintain_frame_rate(self, last_processing_time: float):
        """
        Wait to maintain the frame rate of the episode player.

        :param last_processing_time: The time of the last processing.
        """
        time_diff = time.time() - last_processing_time
        if time_diff < self.time_between_frames.total_seconds():
            time.sleep(self.time_between_frames.total_seconds() - time_diff)


class FileEpisodePlayer(EpisodePlayer):
    def __init__(self, json_file: str, scene_id: int = 1, world: Optional[World] = None,
                 mesh_scale: float = 0.001,
                 time_between_frames: datetime.timedelta = datetime.timedelta(milliseconds=50)):
        """
        Initializes the FAMEEpisodePlayer with the specified json file and scene id.

        :param json_file: The json file that contains the data frames.
        :param scene_id: The scene id.
        :param world: The world that is used to replay the episode.
        :param mesh_scale: The scale of the mesh.
        :param time_between_frames: The time between frames.
        """
        super().__init__(time_between_frames=time_between_frames)
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            self.data_frames = json.load(f)[str(scene_id)]
        self.data_frames = {int(k): v for k, v in self.data_frames.items()}
        self.data_frames = dict(sorted(self.data_frames.items(), key=lambda x: x[0]))
        self.world = world if world is not None else World.current_world
        self.mesh_scale = mesh_scale
        self.copy_model_files_to_world_data_dir()
        self._pause: bool = False

    def copy_model_files_to_world_data_dir(self):
        parent_dir_of_json_file = os.path.abspath(os.path.dirname(self.json_file))
        models_path = os.path.join(parent_dir_of_json_file, "custom", "models")
        # Copy the entire folder and its contents
        shutil.copytree(models_path, self.world.cache_manager.data_directories[0], dirs_exist_ok=True)

    def run(self):
        for frame_id, objects_data in self.data_frames.items():
            self._wait_if_paused()
            last_processing_time = time.time()
            self.process_objects_data(objects_data)
            self._wait_to_maintain_frame_rate(last_processing_time)
            self._ready = True

    def process_objects_data(self, objects_data: dict):
        objects_poses: Dict[Object, Pose] = {}
        for object_id, object_poses_data in objects_data.items():

            # Get the object pose in the map frame
            pose = self.get_pose_and_transform_to_map_frame(object_poses_data[0])

            # Get the object and mesh names
            obj_name = self.get_object_name(object_id)
            mesh_name = self.get_mesh_name(object_id)

            # Create the object if it does not exist in the world and set its pose
            if obj_name not in self.world.get_object_names():
                _ = Object(obj_name, ObjectType.GENERIC_OBJECT, mesh_name,
                           pose=pose, scale_mesh=self.mesh_scale)
            else:
                obj = self.world.get_object_by_name(obj_name)
                objects_poses[obj] = pose
        if len(objects_poses):
            self.world.reset_multiple_objects_base_poses(objects_poses)

    def get_pose_and_transform_to_map_frame(self, object_pose: dict) -> Pose:
        """
        Get the pose of an object and transform it to the map frame.

        :param object_pose: The pose of the object.
        :return: The pose of the object in the map frame.
        """
        position, quaternion = self.get_pose_from_frame_object_data(object_pose)
        return self.transform_pose_to_map_frame(position, quaternion)

    @staticmethod
    def get_pose_from_frame_object_data(object_data: dict) -> Tuple[List[float], List[float]]:
        """
        Get the pose from the frame object data that is extracted from the data file.

        :param object_data: The object data.
        :return: The position and quaternion of the object.
        """
        position = np.array(list(map(float, object_data['t']))) / 1000  # Convert from mm to m
        position = position.tolist()
        rotation = np.array(list(map(float, object_data['R']))).reshape(3, 3)
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation
        quaternion = quaternion_from_matrix(homogeneous_matrix).tolist()
        return position, quaternion

    def transform_pose_to_map_frame(self, position: List[float], quaternion: List[float]) -> Pose:
        """
        Transform the pose of an object to the map frame.

        :param position: The position of the object.
        :param quaternion: The quaternion of the object.
        """
        new_transform = Transform([0, 0, 1],
                                  quaternion_from_matrix(euler_matrix(-np.pi / 2, 0, 0)).tolist(),
                                  child_frame=self.camera_frame_name)
        self.world.local_transformer.update_transforms([new_transform])
        pose = Pose(position, quaternion, frame=self.camera_frame_name)
        return self.world.local_transformer.transform_pose(pose, "map")

    @property
    def camera_frame_name(self) -> str:
        return "episode_camera_frame"

    @staticmethod
    def get_object_name(object_id: str) -> str:
        return f"episode_object_{object_id}"

    @staticmethod
    def get_mesh_name( object_id: str) -> str:
        return f"obj_{object_id:0>6}.ply"