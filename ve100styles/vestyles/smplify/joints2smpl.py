"""Joint positions -> SMPL parameters via SMPLify-3D (from MDM's visualize/simplify_loc2rot.py).

Fitting logic is unchanged; only imports, asset paths and the device argument were adapted.
"""
import h5py
import smplx
import torch

from .. import rotation_conversions as geometry
from . import config
from .smplify import SMPLify3D


class joints2smpl:

    def __init__(self, num_frames, device):
        self.device = torch.device(device)
        self.batch_size = num_frames
        self.num_joints = 22  # HumanML3D skeleton
        self.joint_category = "AMASS"
        self.num_smplify_iters = 150
        self.fix_foot = False
        smplmodel = smplx.create(config.SMPL_MODEL_DIR,
                                 model_type="smpl", gender="neutral", ext="pkl",
                                 batch_size=self.batch_size).to(self.device)

        # mean pose / shape as initialisation
        with h5py.File(config.SMPL_MEAN_FILE, 'r') as file:
            self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
            self.init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)

        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                                 batch_size=self.batch_size,
                                 joints_category=self.joint_category,
                                 num_iters=self.num_smplify_iters,
                                 device=self.device,
                                 use_lbfgs=True)

    def joint2smpl(self, input_joints, init_params=None):
        """input_joints: [nframes, 22, 3] -> (thetas [1, 25, 6, nframes], init params for the next call)."""
        keypoints_3d = torch.as_tensor(input_joints).to(self.device).float()

        if init_params is None:
            pred_betas = self.init_mean_shape
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero
        else:
            pred_betas = init_params['betas']
            pred_pose = init_params['pose']
            pred_cam_t = init_params['cam']

        confidence_input = torch.ones(self.num_joints)
        if self.fix_foot:
            confidence_input[[7, 8, 10, 11]] = 1.5

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
            new_opt_cam_t, new_opt_joint_loss = self.smplify(
                pred_pose.detach(),
                pred_betas.detach(),
                pred_cam_t.detach(),
                keypoints_3d,
                conf_3d=confidence_input.to(self.device),
            )

        thetas = new_opt_pose.reshape(self.batch_size, 24, 3)
        thetas = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(thetas))  # [nframes, 24, 6]
        root_loc = keypoints_3d[:, 0].clone()  # [nframes, 3]
        root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(1)  # [nframes, 1, 6]
        thetas = torch.cat([thetas, root_loc], dim=1).unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 6, nframes]

        return thetas.clone().detach(), {'pose': new_opt_joints[0, :24].flatten().clone().detach(),
                                         'betas': new_opt_betas.clone().detach(),
                                         'cam': new_opt_cam_t.clone().detach()}
