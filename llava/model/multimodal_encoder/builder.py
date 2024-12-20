import os
import dataclasses
import torch
import torch.nn as nn


from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dino_encoder import DINOVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    # if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
    # handles local download of model
    if is_absolute_path_exists:
        if 'openai/clip' in vision_tower or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            if use_s2:
                return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif 'facebook/dino' in vision_tower:
            return DINOVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith('multiple'):
        return MultipleVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            
    raise ValueError(f'Unknown vision tower: {vision_tower}')



vision_tower_paths_dict = {
    'clipL336': '/fsx/wpq/.results/baselines/openai/clip-vit-large-patch14-336',
    'dinov2L': '/fsx/wpq/.results/baselines/facebook/dinov2-large',
}


class MultipleVisionTower(nn.Module):
    def __init__(self, vision_tower_names, args, delay_load=False):
        super().__init__()

        self.vision_tower_names = vision_tower_names.split(':')[-1].split(',')
        self.vision_tower_paths = [vision_tower_paths_dict[x] for x in self.vision_tower_names]
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        vision_towers = nn.ModuleList()
        for vision_tower_path in self.vision_tower_paths:
            vision_tower_cfg = dataclasses.replace(args)
            vision_tower_cfg.mm_vision_tower = vision_tower_path
            vision_towers += [build_vision_tower(vision_tower_cfg, delay_load=delay_load)]
        self.vision_towers = vision_towers

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_names))
            return

        for vision_tower in self.vision_towers:
            vision_tower.load_model(device_map=device_map)

        self.image_processor = [
            x.image_processor for x in self.vision_towers
        ]
        self.vision_tower = [
            x.vision_tower for x in self.vision_towers
        ]

    @property
    def is_loaded(self):
        return all(x.is_loaded for x in self.vision_towers)

    def feature_select(self, image_forward_outs):
        if len(image_forward_outs) != len(self.vision_towers):
            raise ValueError(f'images ({len(image_forward_outs)}) != number of vision towers ({len(self.vision_towers)})')
        image_features_list = []
        for image_forward_out, vision_tower in zip(image_forward_outs, self.vision_towers):
            image_features_list.append(vision_tower.feature_select(image_forward_out))
        return image_features_list

    @torch.no_grad()
    def forward(self, images):
        if len(images) != len(self.vision_towers):
            raise ValueError(f'images ({len(images)}) != number of vision towers ({len(self.vision_towers)})')
        image_features_list = []
        for image, vision_tower in zip(images, self.vision_towers):
            image_features_list.append(vision_tower.forward(image))
        return image_features_list

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_towers[0].dtype

    @property
    def device(self):
        return self.vision_towers[0].device

    @property
    def config(self):
        if self.is_loaded:
            return tuple(x.config for x in self.vision_towers)
        else:
            return tuple(x.cfg_only for x in self.vision_towers)

    @property
    def hidden_size(self):
        return tuple(x.config for x in self.vision_towers)

    @property
    def num_patches_per_side(self):
        return tuple(x.num_patches_per_side for x in self.vision_towers)

    @property
    def num_patches(self):
        return tuple(x.num_patches for x in self.vision_towers)
