import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from typing import Union, Optional, Tuple
import os
from llava.utils import rank0_print
from einops import rearrange
import sys
sys.path.append('CUT3R')
from src.dust3r.model import ARCroco3DStereo
import numpy as np

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    rank0_print("Warning: open3d not found. Point cloud export functionality will be disabled.")
    _OPEN3D_AVAILABLE = False

class Cut3rSpatialConfig(PretrainedConfig):
    model_type = "cut3r_spatial_model"

    def __init__(
        self,
        weights_path="/data/zhanpeng/weight/vlm3r/cut3r/cut3r_512_dpt_4_64.pth",
        spatial_tower_select_feature="patch",
        spatial_tower_select_layer=-1,
        export_point_cloud: bool = False,
        point_cloud_output_dir: str = "point_clouds",
        point_cloud_voxel_size: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights_path = weights_path
        self.spatial_tower_select_feature = spatial_tower_select_feature
        self.spatial_tower_select_layer = spatial_tower_select_layer
        self.export_point_cloud = export_point_cloud
        self.point_cloud_output_dir = point_cloud_output_dir
        self.point_cloud_voxel_size = point_cloud_voxel_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the spatial config dict if we are loading from Cut3rSpatialConfig
        if config_dict.get("model_type") == "cut3r":
            config_dict = config_dict["spatial_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)

class Cut3rSpatialPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Cut3rSpatialConfig
    base_model_prefix = "cut3r"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

def prepare_input(pixel_values):
    pixel_values = nn.functional.interpolate(pixel_values, size=(432, 432), mode='bilinear')
    pixel_values = pixel_values.unsqueeze(1) ## FIXME: the second dimension is the number of frames in one batch
    views = []
    # Assuming pixel_values is (F_max, B, C, H, W)
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 5:
        raise ValueError(f"Expected pixel_values to be a 5D tensor (F, B, C, H, W), got {type(pixel_values)} with shape {getattr(pixel_values, 'shape', 'N/A')}")

    F_max, B, C, H, W = pixel_values.shape
    device = pixel_values.device

    for i in range(F_max):
        current_frame_batch = pixel_values[i] # Shape (B, C, H, W)
        view = {
            "img": current_frame_batch,
            "ray_map": torch.full(
                (
                    B, # Use batch size B
                    6,
                    H, # Use H
                    W, # Use W
                ),
                torch.nan,
            ).to(device),
            "true_shape": torch.tensor([H, W], device=device).expand(B, -1), # Shape (B, 2)
            "idx": i,
            "instance": [str(j) for j in range(B)], # List of B instances
            "camera_pose": torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1), # Shape (B, 4, 4)
            "img_mask": torch.tensor(True, device=device).expand(B), # Shape (B)
            "ray_mask": torch.tensor(False, device=device).expand(B), # Shape (B)
            "update": torch.tensor(True, device=device).expand(B), # Shape (B)
            "reset": torch.tensor(False, device=device).expand(B), # Shape (B)
        }
        views.append(view)
    return views

class Cut3rEncoder(nn.Module):
    def __init__(self, config: Cut3rSpatialConfig, **kwargs):
        super().__init__()
        config.weights_path = "/data/zhanpeng/weight/vlm3r/cut3r/cut3r_512_dpt_4_64.pth"
        self.cut3r = ARCroco3DStereo.from_pretrained(config.weights_path)
        self.cut3r.eval()
        self.config = config
        for param in self.cut3r.parameters():
            param.requires_grad = False

    def export_point_cloud(self, views, ress, point_cloud_output_paths: Optional[list[str]]):
        if not _OPEN3D_AVAILABLE:
            rank0_print("Skipping point cloud export because open3d is not available.")
            return
        if not ress:
            rank0_print("Warning: 'ress' list is empty, skipping point cloud export.")
            return
        if not point_cloud_output_paths:
            rank0_print("Warning: point_cloud_output_paths is not provided. Skipping point cloud export.")
            return

        try:
            # Determine batch decoder_config_dict = decoder_config.to_dict()size (B) from the first frame's output if available
            if 'pts3d_in_other_view' not in ress[0] or ress[0]['pts3d_in_other_view'] is None:
                 rank0_print("Warning: Cannot determine batch size from ress[0]. Skipping point cloud export.")
                 return
            B = ress[0]['pts3d_in_other_view'].shape[0]
            num_frames = len(views)

            if len(point_cloud_output_paths) != B:
                 rank0_print(f"Warning: Number of output paths ({len(point_cloud_output_paths)}) does not match batch size ({B}). Skipping point cloud export.")
                 return

            # Ensure base output directory exists (individual file paths might contain subdirs)
            base_output_dir = os.path.dirname(point_cloud_output_paths[0])
            if base_output_dir:
                os.makedirs(base_output_dir, exist_ok=True)

            for b_idx in range(B):
                pts3d_batch_item = []
                colors_batch_item = []
                valid_frames_for_batch_item = 0

                try:
                    # Ensure the specific directory for this batch item exists
                    item_output_dir = os.path.dirname(point_cloud_output_paths[b_idx])
                    if item_output_dir:
                        os.makedirs(item_output_dir, exist_ok=True)

                    # --- Combine data across frames for the current batch item --- 
                    for i in range(num_frames):
                        # Check if the tensors exist and have the expected batch dimension for this frame
                        if (i < len(ress) and 
                            'pts3d_in_other_view' in ress[i] and 
                            'rgb' in ress[i] and 
                            ress[i]['pts3d_in_other_view'] is not None and 
                            ress[i]['rgb'] is not None and 
                            ress[i]['pts3d_in_other_view'].shape[0] == B and 
                            ress[i]['rgb'].shape[0] == B):
                            
                            # Extract data for the current batch item b_idx from frame i
                            pts3d_batch_item.append(ress[i]["pts3d_in_other_view"][b_idx]) # Shape (H, W, 3)
                            colors_batch_item.append(ress[i]["rgb"][b_idx]) # Shape (H, W, 3)
                            valid_frames_for_batch_item += 1
                        else:
                            rank0_print(f"Warning: Missing, None, or mismatched data in ress[{i}] for batch index {b_idx}. Skipping frame {i} for this item.")
                    # --- End combining data --- 

                    if not pts3d_batch_item: # Skip if no valid frames found for this batch item
                        rank0_print(f"Warning: No valid frame data found for batch item {b_idx}. Skipping point cloud generation.")
                        continue # Skip to the next batch item

                    # Stack tensors from valid frames and reshape
                    # Stacking gives (F_valid, H, W, 3) -> view gives (F_valid*H*W, 3)
                    pts3d_single = torch.stack(pts3d_batch_item, dim=0).reshape(-1, 3).float().cpu().numpy()
                    colors_single_stacked = torch.stack(colors_batch_item, dim=0).reshape(-1, 3)
                    colors_single = (colors_single_stacked * 0.5 + 0.5).float().cpu().numpy() # Apply color transform

                    # Create Open3D point cloud for the item
                    pcd_item = o3d.geometry.PointCloud()
                    pcd_item.points = o3d.utility.Vector3dVector(pts3d_single)
                    pcd_item.colors = o3d.utility.Vector3dVector(colors_single)

                    # Voxel downsampling for the item
                    voxel_size = self.config.point_cloud_voxel_size
                    # rank0_print(f"Original combined point cloud (batch item {b_idx}) has {len(pcd_item.points)} points.")
                    downsampled_pcd_item = pcd_item.voxel_down_sample(voxel_size)
                    # rank0_print(f"Downsampled combined point cloud (batch item {b_idx}) has {len(downsampled_pcd_item.points)} points.")

                    # Save the downsampled point cloud for the item
                    output_path = point_cloud_output_paths[b_idx]
                    o3d.io.write_point_cloud(output_path, downsampled_pcd_item)
                    rank0_print(f"Saved combined and downsampled point cloud for batch item {b_idx} to {output_path}")

                except (KeyError, IndexError, AttributeError, ValueError, RuntimeError, Exception) as item_e:
                    rank0_print(f"Error processing or saving point cloud for batch item {b_idx}: {item_e}")
                    import traceback
                    rank0_print(traceback.format_exc())
                    # Continue to the next batch item

        except (KeyError, IndexError, AttributeError, ValueError, RuntimeError, Exception) as e:
            rank0_print(f"Error during point cloud export setup (batch level): {e}")
            import traceback
            rank0_print(traceback.format_exc())

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None
    ) -> Union[Tuple, BaseModelOutput]:
        
        views = prepare_input(pixel_values=pixel_values)
        shape, feat_ls, pos = self.cut3r._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self.cut3r._init_state(feat[0], pos[0])
        mem = self.cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        # spatial_feat = []
        patch_features = []
        camera_tokens = []
        for i in range(len(views)):
            feat_i = feat[i].to(pixel_values.dtype)
            pos_i = pos[i]
            if self.cut3r.pose_head_flag:
                global_img_feat_i = self.cut3r._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.cut3r.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.cut3r.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec = self.cut3r._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.cut3r.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.cut3r.dec_depth + 1
            head_input = [
                dec[0],
                dec[self.cut3r.dec_depth * 2 // 4][:, 1:],
                dec[self.cut3r.dec_depth * 3 // 4][:, 1:],
                dec[self.cut3r.dec_depth],
            ]
            res = self.cut3r._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            # # select feature
            # if self.config.spatial_tower_select_feature == "patch":
            #     spatial_feat_layer = dec[int(self.config.spatial_tower_select_layer)][:, 1:]
            # elif self.config.spatial_tower_select_feature == "all":
            #     spatial_feat_layer = dec[int(self.config.spatial_tower_select_layer)]
            # else:
            #     raise ValueError(f"Unexpected spatial tower select feature: {self.config.spatial_tower_select_feature}")
            # spatial_feat.append(spatial_feat_layer)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].to(pixel_values.dtype)
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].to(pixel_values.dtype)
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )

            # add camera token
            camera_tokens.append(dec[-1][:, :1].clone())
            # add patch features
            patch_features.append(dec[-1][:, 1:].clone())

        # # for debug - Modified for batch processing
        # if not ress: # Handle empty ress case
        #     print("Warning: ress list is empty, skipping point cloud debug output.")
        #     B_debug = 0
        # else:
        #     # Determine batch size (B) from the first element/frame's output
        #     # Assuming ress[0]['pts3d_in_other_view'] has shape (B, H, W, 3)
        #     B_debug = ress[0]['pts3d_in_other_view'].shape[0]

        # for b_idx in range(B_debug):
        #     pts3d_batch_item = []
        #     colors_batch_item = []
        #     valid_frames_for_batch_item = 0
        #     for i in range(len(views)): # Iterate through frames (F_max)
        #         # Check if the tensors exist and have the expected batch dimension
        #         if (i < len(ress) and 'pts3d_in_other_view' in ress[i] and 'rgb' in ress[i] and
        #             ress[i]['pts3d_in_other_view'].shape[0] == B_debug and ress[i]['rgb'].shape[0] == B_debug):
        #             # Extract data for the current batch item b_idx
        #             pts3d_batch_item.append(ress[i]["pts3d_in_other_view"][b_idx]) # Shape (H, W, 3)
        #             colors_batch_item.append(ress[i]["rgb"][b_idx]) # Shape (H, W, 3)
        #             valid_frames_for_batch_item += 1
        #         else:
        #             print(f"Warning: Missing or mismatched data in ress[{i}] for batch index {b_idx}. Skipping frame {i} for this batch item.")

        #     if not pts3d_batch_item: # Skip if no valid frames found for this batch item
        #         print(f"Warning: No valid frame data found for batch item {b_idx}. Skipping point cloud generation.")
        #         continue

        #     # Concatenate across valid frames for this batch item
        #     # Need to stack along a new dimension (e.g., frame dim) then reshape, or cat along H/W dim if appropriate?
        #     # Original DUST3R might have concatenated along H or W. Let's assume stacking then view is safer.
        #     # Stacking gives (F_valid, H, W, 3) -> view gives (F_valid*H*W, 3)
        #     pts3d_single = torch.stack(pts3d_batch_item, dim=0).view(-1, 3).float().cpu().numpy()
        #     colors_single = torch.stack(colors_batch_item, dim=0).view(-1, 3)
        #     colors_single = (colors_single * 0.5 + 0.5).float().cpu().numpy() # Apply color transform

        #     # 保存为点云
        #     try:
        #         import open3d as o3d
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(pts3d_single)
        #         pcd.colors = o3d.utility.Vector3dVector(colors_single)

        #         # voxelize
        #         voxel_size = 0.05 # Example: 5cm voxel size
        #         # print(f"Original point cloud (batch item {b_idx}) has {len(pcd.points)} points.")

        #         # Apply voxel downsampling
        #         downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        #         # print(f"Downsampled point cloud (batch item {b_idx}) has {len(downsampled_pcd.points)} points.")

        #         # Save the downsampled point cloud
        #         output_filename = f"downsampled_point_cloud_batch_{b_idx}.ply"
        #         o3d.io.write_point_cloud(output_filename, downsampled_pcd)
        #         # print(f"Saved point cloud for batch item {b_idx} to {output_filename}")
        #     except ImportError:
        #         print("Warning: open3d not found. Cannot save debug point cloud.")
        #     except Exception as e_pcd:
        #          print(f"Error processing or saving point cloud for batch item {b_idx}: {e_pcd}")

        # spatial_feat = torch.stack(spatial_feat, dim=0) # [frame, b, token_num, token_dim]
        # spatial_feat = rearrange(spatial_feat, 'frame b token_num token_dim -> (b frame) token_num token_dim')
        # return (spatial_feat, None, None, None)

        # Export point cloud if enabled
        if self.config.export_point_cloud:
            # Pass the specific output paths for this batch
            self.export_point_cloud(views, ress, point_cloud_output_paths)

        # rearrange patch_features, camera_tokens, hidden_states
        patch_features = torch.stack(patch_features, dim=0)
        patch_features = rearrange(patch_features, 'frame batch token_num token_dim -> (batch frame) token_num token_dim')
        camera_tokens = torch.stack(camera_tokens, dim=0)
        camera_tokens = rearrange(camera_tokens, 'frame batch token_num token_dim-> (batch frame) token_num token_dim')

        return (camera_tokens, patch_features)

class Cut3rSpatialTransformer(nn.Module):
    def __init__(self, config: Cut3rSpatialConfig, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = Cut3rEncoder(config=config, **kwargs)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_cloud_output_paths=point_cloud_output_paths
        )

        return encoder_outputs

        # last_hidden_state = encoder_outputs[0]

        # if not return_dict:
        #     return (last_hidden_state, None) + encoder_outputs[1:]

        # return BaseModelOutputWithPooling(
        #     last_hidden_state=last_hidden_state,
        #     pooler_output=None,
        #     hidden_states=None,
        #     attentions=None,
        # )

class Cut3rSpatialModel(Cut3rSpatialPreTrainedModel):
    config_class = Cut3rSpatialConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["Cut3rSpatialEncoderLayer"]

    def __init__(self, config: Cut3rSpatialConfig, **kwargs):
        super().__init__(config)

        self.spatial_model = Cut3rSpatialTransformer(config, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config: Cut3rSpatialConfig, **kwargs):
        model = cls(config=config, **kwargs)
        return model

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.spatial_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_cloud_output_paths=point_cloud_output_paths
        )

class Cut3rSpatialTower(nn.Module):
    def __init__(self, spatial_tower, spatial_tower_cfg, delay_load=True):
        super().__init__()

        self.is_loaded = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        vlm_3r_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        dynamic_weights_path = os.path.join(vlm_3r_root, 'CUT3R', 'src', 'cut3r_512_dpt_4_64.pth')

        self.config = Cut3rSpatialConfig(
            weights_path=dynamic_weights_path,
            spatial_tower_select_feature=getattr(spatial_tower_cfg, 'spatial_tower_select_feature', 'patch'),
            spatial_tower_select_layer=getattr(spatial_tower_cfg, 'spatial_tower_select_layer', -1)
        )

        self.spatial_tower_name = spatial_tower

        if not delay_load:
            rank0_print(f"Loading spatial tower: {spatial_tower}")
            self.load_model()
        elif getattr(spatial_tower_cfg, "unfreeze_mm_spatial_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `unfreeze_mm_spatial_tower`: True.")
            self.load_model()
        elif hasattr(spatial_tower_cfg, "mm_tunable_parts") and "mm_spatial_tower" in spatial_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `mm_tunable_parts` contains `mm_spatial_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.spatial_tower_name))
            return

        self.spatial_tower = Cut3rSpatialModel.from_pretrained(self.spatial_tower_name, config=self.config, device_map=device_map)

        self.spatial_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.spatial_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.last_hidden_state.to(image.dtype)
                # assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            image_features = self.spatial_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # image_features = image_features_forward_out.to(images.dtype)
            # assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.spatial_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.spatial_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size