"""
Custom inference script for Grasp-as-You-Say
Input: custom point cloud (.ply) + language instruction
Output: visualized grasp results (IDGC + QGC)
"""
import argparse
import json
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader

sys.path.append("./")
from datasets.task_dex_datasets import DgnBase
from model import build_model
from utils.config_utils import EasyConfig
from model.utils.hand_model import HandModel


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomDataset(DgnBase):
    """Custom dataset for single object inference"""
    def __init__(self, obj_pc, cate_id, guidance, action_id, rotation_type="euler", norm_type="minmax11"):
        super().__init__(rotation_type=rotation_type, norm_type=norm_type)
        self.obj_pc = torch.tensor(obj_pc).float()
        self.cate_id = cate_id
        self.guidance = guidance
        self.action_id = action_id
        self.rotation_type = rotation_type

    def __len__(self):
        return 1

    def __getitem__(self, index):
        intend_id = torch.tensor([int(self.action_id)]) - 1
        intend_vector = torch.zeros(4)
        intend_vector[int(intend_id.item())] = 1
        cls_vector = torch.zeros(len(self.ALL_CAT))
        if self.cate_id in self.ALL_CAT_DCIT:
            cls_vector[self.ALL_CAT_DCIT[self.cate_id]] = 1

        # Create a dummy norm_pose for inference (will be ignored during sampling)
        # The model will generate new poses from scratch
        if self.rotation_type == "euler":
            rotation_dim = 3
        elif self.rotation_type == "quaternion":
            rotation_dim = 4
        elif self.rotation_type == "rotation_6d":
            rotation_dim = 6
        else:
            rotation_dim = 3

        # norm_pose: translation(3) + qpos(22) + rotation(rotation_dim)
        norm_pose = torch.zeros(3 + 22 + rotation_dim)

        # hand_model_pose: translation(3) + axis_angle(3) + qpos(22) = 28
        # This is a dummy pose for inference (needed for matching algorithm)
        hand_model_pose = torch.zeros(28)

        sample = {
            "cate_id": self.cate_id,
            "guidance": self.guidance,
            "cls_vector": cls_vector,
            "obj_pc": self.obj_pc,
            "obj_id": "custom_obj",
            "intend_id": self.action_id,
            "intend_vector": intend_vector,
            "rotation_type": self.rotation_type,
            "norm_pose": norm_pose,  # Required for model inference
            "hand_model_pose": hand_model_pose,  # Required for matching
        }
        return sample

    @staticmethod
    def collate_fn(batch):
        input_dict = {}
        for k in batch[0]:
            if isinstance(batch[0][k], torch.Tensor):
                try:
                    input_dict[k] = torch.stack([sample[k] for sample in batch])
                except:
                    input_dict[k] = [sample[k] for sample in batch]
            else:
                input_dict[k] = [sample[k] for sample in batch]
        return input_dict


class CustomInference:
    def __init__(self, idgc_checkpoint, qgc_checkpoint, device="cuda:0"):
        self.device = torch.device(device)
        setup_seed(3407)

        # Resolve checkpoint paths
        idgc_checkpoint = self._resolve_checkpoint_path(idgc_checkpoint, "idgc")
        qgc_checkpoint = self._resolve_checkpoint_path(qgc_checkpoint, "qgc")

        # Load IDGC model
        print("Loading IDGC model...")
        self.idgc_cfg = EasyConfig()
        self.idgc_cfg.load("./config/idgc.yaml")
        self.idgc_cfg.model.checkpoint_path = idgc_checkpoint
        self.idgc_cfg.model.aux_outputs = False
        self.idgc_model = self._load_model(self.idgc_cfg.model)
        print(f"IDGC model loaded from {idgc_checkpoint}")

        # Load QGC model
        print("Loading QGC model...")
        self.qgc_cfg = EasyConfig()
        self.qgc_cfg.load("./config/qgc.yaml")
        self.qgc_cfg.model.checkpoint_path = qgc_checkpoint
        self.qgc_cfg.model.aux_outputs = False
        self.qgc_model = self._load_model(self.qgc_cfg.model)
        print(f"QGC model loaded from {qgc_checkpoint}")

        # Load hand model for visualization
        print("Loading hand model...")
        self.hand_model = self._load_hand_model()

    def _resolve_checkpoint_path(self, checkpoint_path, model_type):
        """
        Resolve checkpoint path with smart matching.

        Args:
            checkpoint_path: Can be:
                - Full path to .pth file
                - Directory path (will find latest checkpoint)
                - "latest" (will find in ./Experiments/{model_type}/)
                - Epoch number like "200" (will find epoch200_*_latest.pth)
        """
        import glob

        # If it's already a valid file, return it
        if osp.isfile(checkpoint_path) and checkpoint_path.endswith('.pth'):
            return checkpoint_path

        # Determine search directory
        if osp.isdir(checkpoint_path):
            search_dir = checkpoint_path
        elif checkpoint_path.lower() == "latest" or checkpoint_path.isdigit():
            search_dir = f"./Experiments/{model_type}"
        else:
            # Assume it's a path that doesn't exist yet
            return checkpoint_path

        # Find checkpoint files
        if checkpoint_path.isdigit():
            # Search for specific epoch
            pattern = osp.join(search_dir, f"epoch{checkpoint_path}_*_latest.pth")
            matches = glob.glob(pattern)
            if matches:
                print(f"Found checkpoint: {matches[0]}")
                return matches[0]
            else:
                raise FileNotFoundError(f"No checkpoint found for epoch {checkpoint_path} in {search_dir}")
        else:
            # Search for latest checkpoint
            pattern = osp.join(search_dir, "*_latest.pth")
            matches = glob.glob(pattern)
            if matches:
                # Sort by epoch number
                matches.sort(key=lambda x: int(x.split('epoch')[1].split('_')[0]), reverse=True)
                print(f"Found latest checkpoint: {matches[0]}")
                return matches[0]
            else:
                raise FileNotFoundError(f"No checkpoint found in {search_dir}")

    def _load_model(self, model_cfg):
        model = build_model(model_cfg)
        ckpt = torch.load(model_cfg.checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _load_hand_model(self):
        class HandConfig:
            def __init__(self):
                self.mjcf_path = "./data/mjcf/shadow_hand.xml"
                self.mesh_path = "./data/mjcf/meshes"
                self.n_surface_points = 1024
                self.contact_points_path = "./data/mjcf/contact_points.json"
                self.penetration_points_path = "./data/mjcf/penetration_points.json"
                self.fingertip_points_path = "./data/mjcf/fingertip.json"

        hand_config = HandConfig()
        # Use CPU for hand model to avoid device mismatch issues
        hand_model = HandModel(hand_config, "cpu")
        return hand_model

    def _recursive_to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, list):
            return [self._recursive_to_device(d) for d in data]
        elif isinstance(data, tuple):
            return tuple([self._recursive_to_device(d) for d in data])
        elif isinstance(data, dict):
            return {k: self._recursive_to_device(v) for k, v in data.items()}
        else:
            return data

    def load_point_cloud(self, mesh_path, n_points=4096):
        """Load mesh from .obj or .ply file and sample point cloud"""
        print(f"Loading mesh from {mesh_path}...")
        loaded = trimesh.load(mesh_path, process=False, force="mesh", skip_materials=True)

        # Handle different trimesh return types (Trimesh, Scene, PointCloud, etc.)
        if isinstance(loaded, trimesh.Scene):
            # If it's a scene, extract the geometry
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in loaded.geometry.values()))
        elif isinstance(loaded, trimesh.PointCloud):
            # If it's a point cloud, convert to mesh using convex hull or ball pivoting
            print("Loaded data is a point cloud. Converting to mesh using convex hull...")
            vertices = np.array(loaded.vertices)
            try:
                # Try to create a convex hull mesh
                mesh = trimesh.convex.convex_hull(vertices)
                print(f"Created convex hull mesh from point cloud")
            except:
                # If convex hull fails, create a simple visualization mesh (spheres at points)
                print("Convex hull failed. Creating simple sphere representation...")
                # Use a subset of points for visualization
                sample_indices = np.random.choice(len(vertices), min(100, len(vertices)), replace=False)
                spheres = []
                for idx in sample_indices:
                    sphere = trimesh.primitives.Sphere(radius=0.002, center=vertices[idx])
                    spheres.append(sphere)
                mesh = trimesh.util.concatenate(spheres)
        elif hasattr(loaded, 'vertices') and hasattr(loaded, 'faces'):
            mesh = loaded
        else:
            # Last resort: try to extract vertices and create convex hull
            vertices = np.array(loaded.vertices) if hasattr(loaded, 'vertices') else np.array(loaded)
            mesh = trimesh.convex.convex_hull(vertices)

        # Ensure it's a Trimesh object
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            else:
                raise ValueError("Could not convert loaded data to mesh")

        # Center the mesh
        bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
        mesh.vertices = mesh.vertices - bbox_center

        # Sample points from the mesh
        if hasattr(mesh, 'sample'):
            point_cloud = mesh.sample(n_points)
        else:
            # If sampling is not available, use vertices directly
            vertices = np.array(mesh.vertices)
            if len(vertices) > n_points:
                indices = np.random.choice(len(vertices), n_points, replace=False)
                point_cloud = vertices[indices]
            else:
                # Repeat points if we have fewer than n_points
                indices = np.random.choice(len(vertices), n_points, replace=True)
                point_cloud = vertices[indices]

        print(f"Loaded point cloud with {len(point_cloud)} points")
        print(f"Point cloud bounds: min={point_cloud.min(axis=0)}, max={point_cloud.max(axis=0)}")
        print(f"Object mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return point_cloud, mesh

    @torch.no_grad()
    def infer_idgc(self, data_dict):
        """Stage 1: IDGC inference"""
        print("\n=== Stage 1: IDGC Inference ===")
        data_dict = self._recursive_to_device(data_dict)

        _, _, predictions = self.idgc_model.forward_test(data_dict)
        raw_predictions = predictions["outputs"]["hand_model_pose"]  # (num_queries, 28)
        matched_predictions = predictions["outputs"]["matched"]["hand_model_pose"]  # (K, 28)

        print(f"IDGC generated {raw_predictions.shape[0]} grasp candidates")
        print(f"IDGC matched {matched_predictions.shape[0]} grasps")

        return {
            "raw_predictions": raw_predictions.cpu(),
            "matched_predictions": matched_predictions.cpu(),
        }

    def _normalize_pose(self, pose):
        """
        Convert raw pose (28D) to normalized pose for QGC

        Input format:
            pose: [B, 28] = translation(3) + axis_angle(3) + qpos(22)

        Output format (depends on rotation_type from config):
            - rotation_type="euler": [B, 28] = norm_translation(3) + norm_qpos(22) + norm_euler(3)
            - rotation_type="quaternion": [B, 29] = norm_translation(3) + norm_qpos(22) + quaternion(4)
            - rotation_type="rotation_6d": [B, 31] = norm_translation(3) + norm_qpos(22) + rotation_6d(6)
        """
        import pytorch3d.transforms as T
        from datasets.task_dex_datasets import DgnBase

        # Extract components
        hand_translation = pose[:, :3]  # (B, 3)
        hand_axis_angle = pose[:, 3:6]  # (B, 3)
        hand_qpos = pose[:, 6:]  # (B, 22)

        # Get normalization factors from IDGC config
        norm_type = self.idgc_cfg.data.train.norm_type
        if "minmax" in norm_type:
            factor = DgnBase.factor_minmax
        else:
            factor = DgnBase.factor_meastd

        # Normalize qpos
        norm_qpos = self._norm_by_type(hand_qpos, factor[6:], norm_type)

        # Convert axis-angle to euler then to rotation representation
        hand_euler = torch.flip(
            T.matrix_to_euler_angles(T.axis_angle_to_matrix(hand_axis_angle), "ZYX"),
            dims=[-1]
        )

        # Transform rotation based on rotation type
        rotation_type = self.idgc_cfg.data.train.rotation_type
        if rotation_type == "euler":
            norm_rotation = hand_euler
            norm_rotation = self._norm_by_type(norm_rotation, factor[3:6], norm_type)
        elif rotation_type == "quaternion":
            norm_rotation = T.matrix_to_quaternion(T.euler_angles_to_matrix(hand_euler, "XYZ"))
        elif rotation_type == "rotation_6d":
            norm_rotation = T.matrix_to_rotation_6d(T.euler_angles_to_matrix(hand_euler, "XYZ"))
        else:
            norm_rotation = hand_euler

        # Normalize translation
        norm_translation = self._norm_by_type(hand_translation, factor[:3], norm_type)

        # Concatenate: translation(3) + qpos(22) + rotation(?)
        norm_pose = torch.cat([norm_translation, norm_qpos, norm_rotation], dim=-1)

        return norm_pose

    def _norm_by_type(self, input_tensor, minmax, norm_type):
        """Normalize input based on type"""
        if norm_type == "minmax11":
            return 2 * (input_tensor - minmax[:, 0]) / (minmax[:, 1] - minmax[:, 0]) - 1
        elif norm_type == "meastd11":
            means = minmax[:, 0]
            stds = minmax[:, 1]
            return (input_tensor - means) / (2 * stds)
        else:
            return input_tensor

    @torch.no_grad()
    def infer_qgc(self, data_dict, idgc_result):
        """Stage 2: QGC refinement"""
        print("\n=== Stage 2: QGC Refinement ===")

        matched_poses = idgc_result["matched_predictions"]  # [K, 28]
        K = matched_poses.shape[0]

        all_refined = []

        # Process each IDGC matched grasp individually
        for i in range(K):
            # Single grasp
            single_pose = matched_poses[i:i+1].to(self.device)  # [1, 28]

            # Normalize
            single_norm_pose = self._normalize_pose(single_pose.cpu()).to(self.device)  # [1, 28]

            # Prepare single sample data dictionary
            # Copy metadata fields from original data_dict
            # Note: Do NOT provide "matched" field - it will be auto-generated by the matcher in forward_test
            single_data_dict = {
                "obj_pc": data_dict["obj_pc"],          # [1, 4096, 3]
                "coarse_pose": single_pose,             # [1, 28]
                "coarse_norm_pose": single_norm_pose,   # [1, 28]
                "norm_pose": single_norm_pose,          # [1, 28] - target placeholder for matching
                "hand_model_pose": single_pose,         # [1, 28] - target placeholder for matching
                # Metadata fields required by matcher and loss functions
                "rotation_type": data_dict["rotation_type"],  # List[str]
                "cate_id": data_dict["cate_id"],              # List[str]
                "guidance": data_dict["guidance"],            # List[str]
                "obj_id": data_dict["obj_id"],                # List[str]
                "intend_id": data_dict["intend_id"],          # List[str]
                # Note: hand_pc will be auto-generated in QGC forward_test
            }

            # Transfer to device
            single_data_dict = self._recursive_to_device(single_data_dict)

            # Call QGC
            _, _, predictions = self.qgc_model.forward_test(single_data_dict)
            refined = predictions["outputs"]["matched"]["hand_model_pose"]  # [M, 28]

            all_refined.append(refined.cpu())

        # Merge all results
        all_refined = torch.cat(all_refined, dim=0) if all_refined else torch.empty(0, 28)

        print(f"QGC refined {all_refined.shape[0]} grasps (from {K} IDGC outputs)")

        return {
            "refined_predictions": all_refined,
        }

    def visualize_results(self, obj_mesh, idgc_result, qgc_result, save_dir, max_vis=5):
        """Visualize grasp results"""
        print("\n=== Visualization ===")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Object mesh info: {len(obj_mesh.vertices)} vertices, {len(obj_mesh.faces)} faces")

        # Visualize IDGC results
        print("Visualizing IDGC results...")
        idgc_predictions = idgc_result["matched_predictions"].cpu()
        n_vis = min(max_vis, idgc_predictions.shape[0])

        for i in range(n_vis):
            pose = idgc_predictions[i:i+1]
            hand = self.hand_model(pose, with_meshes=True)
            hand_vertices = hand['vertices'][0].detach().cpu().numpy()
            hand_faces = hand['faces'].detach().cpu().numpy()

            # Manually combine meshes by concatenating vertices and adjusting face indices
            obj_vertices = obj_mesh.vertices.copy()
            obj_faces = obj_mesh.faces.copy()

            # Offset hand face indices by the number of object vertices
            hand_faces_offset = hand_faces + len(obj_vertices)

            # Combine vertices and faces
            combined_vertices = np.vstack([obj_vertices, hand_vertices])
            combined_faces = np.vstack([obj_faces, hand_faces_offset])

            combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)

            save_path = osp.join(save_dir, f"idgc_grasp_{i+1}.obj")
            combined_mesh.export(save_path)
            print(f"  Saved: {save_path} (obj: {len(obj_vertices)}v, hand: {len(hand_vertices)}v, total: {len(combined_vertices)}v)")

        # Visualize QGC results
        print("Visualizing QGC refined results...")
        qgc_predictions = qgc_result["refined_predictions"].cpu()
        n_vis = min(max_vis, qgc_predictions.shape[0])

        for i in range(n_vis):
            pose = qgc_predictions[i:i+1]
            hand = self.hand_model(pose, with_meshes=True)
            hand_vertices = hand['vertices'][0].detach().cpu().numpy()
            hand_faces = hand['faces'].detach().cpu().numpy()

            # Manually combine meshes by concatenating vertices and adjusting face indices
            obj_vertices = obj_mesh.vertices.copy()
            obj_faces = obj_mesh.faces.copy()

            # Offset hand face indices by the number of object vertices
            hand_faces_offset = hand_faces + len(obj_vertices)

            # Combine vertices and faces
            combined_vertices = np.vstack([obj_vertices, hand_vertices])
            combined_faces = np.vstack([obj_faces, hand_faces_offset])

            combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)

            save_path = osp.join(save_dir, f"qgc_refined_grasp_{i+1}.obj")
            combined_mesh.export(save_path)
            print(f"  Saved: {save_path} (obj: {len(obj_vertices)}v, hand: {len(hand_vertices)}v, total: {len(combined_vertices)}v)")

        print(f"\nAll visualizations saved to: {save_dir}")

    def infer(self, mesh_path, guidance, cate_id, action_id, save_dir, num_samples=1):
        """End-to-end inference pipeline"""
        print("=" * 80)
        print("Custom Grasp-as-You-Say Inference")
        print("=" * 80)
        print(f"Input mesh: {mesh_path}")
        print(f"Category: {cate_id}")
        print(f"Action: {action_id}")
        print(f"Guidance: {guidance}")
        print("=" * 80)

        # Load mesh and sample point cloud
        point_cloud, obj_mesh = self.load_point_cloud(mesh_path)

        # Create custom dataset
        dataset = CustomDataset(
            obj_pc=point_cloud,
            cate_id=cate_id,
            guidance=guidance,
            action_id=action_id,
            rotation_type=self.idgc_cfg.data.train.rotation_type,
            norm_type=self.idgc_cfg.data.train.norm_type,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        # Get data
        data_dict = next(iter(dataloader))

        # Stage 1: IDGC
        idgc_result = self.infer_idgc(data_dict)

        # Stage 2: QGC
        qgc_result = self.infer_qgc(data_dict, idgc_result)

        # Visualize
        self.visualize_results(obj_mesh, idgc_result, qgc_result, save_dir)

        # Save results
        results = {
            "mesh_path": mesh_path,
            "category": cate_id,
            "action_id": action_id,
            "guidance": guidance,
            "idgc_grasps": idgc_result["matched_predictions"].numpy().tolist(),
            "qgc_refined_grasps": qgc_result["refined_predictions"].numpy().tolist(),
        }

        result_path = osp.join(save_dir, "inference_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {result_path}")

        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Custom inference for Grasp-as-You-Say")
    parser.add_argument("--mesh_path", "--pc_path", type=str, required=True, dest="mesh_path",
                        help="Path to mesh file (.obj or .ply). Point cloud will be sampled from the mesh.")
    parser.add_argument("--guidance", type=str, required=True,
                        help="Language guidance, e.g., 'grasp the handle to use it'")
    parser.add_argument("--cate_id", type=str, default="screwdriver",
                        help="Object category (see ALL_CAT in task_dex_datasets.py)")
    parser.add_argument("--action_id", type=str, default="0001",
                        choices=["0001", "0002", "0003", "0004"],
                        help="Action ID: 0001=use, 0002=hold, 0003=lift, 0004=hand_over")
    parser.add_argument("--idgc_checkpoint", type=str, default="latest",
                        help="IDGC checkpoint: 'latest', epoch number (e.g., '200'), directory path, or full .pth path")
    parser.add_argument("--qgc_checkpoint", type=str, default="latest",
                        help="QGC checkpoint: 'latest', epoch number (e.g., '100'), directory path, or full .pth path")
    parser.add_argument("--save_dir", type=str, default="./custom_inference_results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize inference engine
    inference = CustomInference(
        idgc_checkpoint=args.idgc_checkpoint,
        qgc_checkpoint=args.qgc_checkpoint,
        device=args.device,
    )

    # Run inference
    results = inference.infer(
        mesh_path=args.mesh_path,
        guidance=args.guidance,
        cate_id=args.cate_id,
        action_id=args.action_id,
        save_dir=args.save_dir,
    )

    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
