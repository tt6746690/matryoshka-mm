#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import torch.nn.functional as F
import numpy as np


try:
    from rosemary import parse_kv_from_string
except:
    pass



class DenseGatingNetwork(torch.nn.Module):
    """A simple mean-pooling gating network for selecting experts.
    reference: https://github.com/facebookresearch/fairseq/blob/main/examples/translation_moe/translation_moe_src/mean_pool_gating_network.py
    """

    def __init__(self, embed_dim, num_experts, dropout=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        self.fc1 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.fc2 = torch.nn.Linear(embed_dim, num_experts)

    def get_dtype(self):
        return self.fc1.weight.dtype

    def forward(self, x):
        # x: (B, D)
        x = torch.tanh(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        # (B, K) where K is number of experts
        p = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32) # bfloat16 -> float32
        return p


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

        if hasattr(config, "config"):
            self.initialize_additional_modules(config.config)
        else:
            self.use_alternative = False


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_additional_modules(self, model_config):
        """
            call the corresponding init function, 
                e.g., if `model_config.projection_type` is "v1", then calls 
                    `self.initialize_additional_modules_v1()`

            assumes `model_config` contains ["use_alternative", "projection_type"]
        """
        self.use_alternative = model_config["use_alternative"]
        if not self.use_alternative:
            return
        self.projection_type = model_config.get('projection_type', 'v4')

        method_name = f"initialize_additional_modules_{self.projection_type}"
        if getattr(self, method_name, None) is None:
            print(f"[MetaModel.initialize_additional_modules] {method_name} not implemented.")
        else:
            getattr(self, method_name)(model_config)

    def initialize_additional_modules_v4(self, model_config):
        """ mixture of experts with matryoshka VLM """
        if self.is_m3_moe:
            kvs = parse_kv_from_string(model_config['moe'])

            model_type = kvs['t']
            if model_type == 'dense':
                feature_type = kvs['ft']
                if feature_type in ('cls', 'clslast', 'patchavgpool', 'poolout'):
                    # embed_dim = self.get_vision_tower().config.hidden_size # not yet loaded
                    embed_dim = self.config.mm_hidden_size
                elif feature_type in ('attnqk', 'attnkk'):
                    # embed_dim = self.get_vision_tower().num_patches
                    embed_dim = 576 # hard code for now.
                else:
                    raise ValueError(f"[initialize_additional_modules_v4] feature_type={feature_type} cannot determine `embed_dim`.")
                self.router = DenseGatingNetwork(
                    embed_dim=embed_dim,
                    num_experts=len(self.tokscale_list),
                    dropout=kvs.get('dropout', None),
                )
            else:
                raise ValueError(f'[initialize_additional_modules_v4] model_type={model_type} not impl.')

    @property
    def tokscale_list(self):
        return eval(parse_kv_from_string(self.config.config['matryoshka_vis_token_scale'])['numtoks']) \
            if self.is_m3_moe else []

    @property
    def is_m3(self):
        return self.use_alternative and \
            self.projection_type == 'v4' and \
            hasattr(self.config, 'config') and \
            self.config.config.get('matryoshka_vis_token_scale', None) is not None

    @property
    def is_m3_moe(self):
        return self.is_m3 and self.config.config.get('moe', None) is not None



def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images_with_attn(self, images):
        # images: (B, 3, H, W)
        vision_tower = self.get_model().get_vision_tower().vision_tower

        outputs = {}
        def hook_k(module, input, output):
            outputs['k'] = output

        def hook_q(module, input, output):
            outputs['q'] = output

        #set hooks for extracting desired layer's k and q. 23 corresponds to the last layer
        hook_handle_k = vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        # forward
        image_forward_outs = vision_tower(
            images.half().to(device=vision_tower.device, dtype=vision_tower.dtype), 
            output_hidden_states=True)
        # (B, 576+1, D)
        cls_patch = image_forward_outs.hidden_states[self.get_model().get_vision_tower().select_layer]
        # (B, N, D)
        patch = cls_patch[:, 1:, :]
        # (B, D)
        cls = cls_patch[:, 0, :]
        # (B, D)
        patchavgpool = cls_patch[:, 1:, :].mean(1)
        # (B, D)
        cls_last = image_forward_outs.hidden_states[-1][:, 0, :]
        # (B, D)
        pooled_output = image_forward_outs.pooler_output

        #extract desired layer's k and q and remove hooks; calculate attention
        k = outputs["k"]
        q = outputs["q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        # (B, N) where N=576
        D = cls_patch.shape[-1]
        attn_qk = (q[:, :1, :] @ k[:, 1:, :].transpose(-2, -1)).squeeze(1) * D ** -0.5
        attn_qk = torch.nn.functional.softmax(attn_qk, dim=-1)
        attn_kk = (k[:, :1, :] @ k[:, 1:, :].transpose(-2, -1)).squeeze(1) * D ** -0.5
        attn_kk = torch.nn.functional.softmax(attn_kk, dim=-1)

        return {
            "patch": patch,
            "cls": cls,
            'clslast': cls_last,
            "patchavgpool": patchavgpool,
            'poolout': pooled_output,
            'attnqk': attn_qk,
            'attnkk': attn_kk,
        }

    def encode_images_original(self, images):
        patch = self.get_model().get_vision_tower()(images)
        return {
            'patch': patch,
        }

    def encode_images(self, images):
        if self.get_model().is_m3:
            return self.encode_images_with_attn(images)
        else:
            return self.encode_images_original(images)

    def project(self, images, matryoshka_vis_token_scale=None):
        """
            call the corresponding `project` function, 
                e.g., if `self.get_model().projection_type` is "v1", then calls 
                    `self.project_v1()`
        """        
        encode_images_output = self.encode_images(images)
        image_features = encode_images_output['patch'] # (B, L, D)

        projector_loc = self.get_model().config.config.get('projector_loc', 'after_vision_tower') \
            if hasattr(self.get_model().config, 'config') else 'after_vision_tower'
        if projector_loc == 'after_vision_tower':
            image_features = self.get_model().mm_projector(image_features)

        gating_prob = self.router_forward(encode_images_output)

        if self.get_model().use_alternative:
            projection_type = self.get_model().projection_type
            method_name = f"project_{projection_type}"
            if not getattr(self, method_name):
                raise ValueError(f"[LlavaMetaForCausalLM.project] {method_name} not supported.")
            if (projection_type == 'v4' and not matryoshka_vis_token_scale) or (projection_type != 'v4' and matryoshka_vis_token_scale):
                raise ValueError('Only use `matryoshka_vis_token_scale` when `projection_type="v4"`.')
            
            if projection_type == "v4":
                image_features = getattr(self, method_name)(image_features, matryoshka_vis_token_scale=matryoshka_vis_token_scale, gating_prob=gating_prob)
            else:
                image_features = getattr(self, method_name)(image_features)
        
        if projector_loc == 'after_pooling':
            image_features = self.get_model().mm_projector(image_features)

        return {
            'image_features': image_features,
            'gating_prob': gating_prob,
        }

    def project_v4(self, image_features, matryoshka_vis_token_scale=None, gating_prob=None):

        if matryoshka_vis_token_scale == '':
            return image_features

        H = W = int(self.get_model().get_vision_tower().config.image_size / self.get_model().get_vision_tower().config.patch_size)
        kvs = parse_kv_from_string(matryoshka_vis_token_scale)

        if kvs['ver'] == 'v0': 
            if kvs['numtoks'] == 'gateprobargmax':
                if gating_prob is None:
                    raise ValueError(f'[LlavaMetaForCausalLM.project_v4] requires `gating_prob` to select the right token scale for matryoshka_vis_token_scale={matryoshka_vis_token_scale}')
                if image_features.shape[0] != 1:
                    raise ValueError(f'[LlavaMetaForCausalLM.project_v4] only support batch_size=1 for matryoshka_vis_token_scale={matryoshka_vis_token_scale} but got {image_features.shape[0]}. This is ok since only used during inference.')
                numtoks_idx = torch.argmax(gating_prob.squeeze(0)).item()
                numtoks = self.get_model().tokscale_list[numtoks_idx]
            else:
                numtoks = int(kvs['numtoks'])

            B, H_W, D = image_features.shape
            reshaped_tensor = image_features.view(B, H, W, D)
            # (B, D, H, W) e.g., (B, 4096, 24, 24)
            reshaped_tensor = reshaped_tensor.permute(0, 3, 1, 2)
            h = w = int( math.sqrt(numtoks) )
            assert(h*w == numtoks)
            # (B, D, h, w)
            pooled_tensor = torch.nn.functional.adaptive_avg_pool2d(reshaped_tensor, (h, w))
            # (B, h, w, D)
            image_features = pooled_tensor.permute(0, 2, 3, 1)
            # (B, numtoks, D)
            image_features = image_features.reshape(B, -1, D)
        else:
            raise ValueError(f"[LlavaMetaForCausalLM.project_v4] {kvs['ver']} not implemented.")

        return image_features

    def router_forward(self, encode_images_output):
        if self.get_model().is_m3_moe:
            kvs = parse_kv_from_string(self.get_model().config.config['moe'])
            feature_type = kvs['ft']
            if feature_type in encode_images_output:
                router_input = encode_images_output[feature_type]
            else:
                raise ValueError(f'feature_type={feature_type} cannot be retrieved from `encode_images_output`')
            # (B, K) float16 -> bfloat16
            gating_prob = self.get_model().router(router_input)
        else:
            gating_prob = None
        return gating_prob
    
    def matryoshka_vis_token_process(self, image_features, matryoshka_vis_token_scale):
        N, H_W, C = image_features.shape
        H = W = int(H_W ** 0.5)
        reshaped_tensor = image_features.view(N, H, W, C)
        reshaped_tensor = reshaped_tensor.permute(0, 3, 1, 2)
        pool_size = stride = int( np.sqrt(H_W / matryoshka_vis_token_scale) )
        pooled_tensor = F.avg_pool2d(reshaped_tensor, kernel_size=pool_size, stride=stride)
        image_features = pooled_tensor.permute(0, 2, 3, 1)
        image_features = image_features.reshape(N, -1, C)
        # print('image_features.shape :', image_features.shape)
        return image_features
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, matryoshka_vis_token_scale = None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # images: (B, 3, H, W)
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            project_outputs = self.project(concat_images, matryoshka_vis_token_scale=matryoshka_vis_token_scale)
            image_features = project_outputs['image_features']
            gating_prob = project_outputs['gating_prob'] # wpq: most likely wrong .. verify this is as intended.
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        # height = width = self.get_vision_tower().num_patches_per_side 
                        height = width = int(np.sqrt(base_image_feature.shape[0]))
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            project_outputs = self.project(images, matryoshka_vis_token_scale=matryoshka_vis_token_scale)
            image_features = project_outputs['image_features']
            gating_prob = project_outputs['gating_prob']
                
                
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, gating_prob

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
