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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

try:
    from rosemary import parse_kv_from_string, create_string_from_kv
except:
    pass


@dataclass
class CausalLMOutputWithPastWithGatingProb(CausalLMOutputWithPast):
    gating_prob: Optional[torch.FloatTensor] = None


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward_single_matryoshka(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        matryoshka_vis_token_scale: Optional[int] = None,
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                gating_prob,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                matryoshka_vis_token_scale = matryoshka_vis_token_scale,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()


        #### wpq: 
        if gating_prob is not None:
            kvs = parse_kv_from_string(matryoshka_vis_token_scale)
            if kvs['ver'] == 'v0':
                if kvs['numtoks'] == 'gateprobargmax':
                    # reached only during inference, and `gating_prob` not useful after this point
                    # since it only participates in weighting the loss.
                    gating_prob_k = None
                else:
                    # reached only during training
                    tokscale_list = self.get_model().tokscale_list
                    k = tokscale_list.index(kvs['numtoks'])
                    # (B, K) -> (B,)
                    gating_prob_k = gating_prob[:, k]
        else:
            gating_prob_k = None
        ####

        if gating_prob_k is None:
            # typical LM loss
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
        else:
            # LM loss weighted by gating prob
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                # (B, seq_len-1, vocab_size)
                shift_logits = logits[..., :-1, :].contiguous()
                # (B, seq_len-1)
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct_noreduce = CrossEntropyLoss(reduction='none')
                L = logits.shape[1]-1
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                gating_prob_k = gating_prob_k.to(shift_logits.device)
                losses = loss_fct_noreduce(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                # (B, seq_len-1)
                losses = losses.view(-1, L)
                valid_mask = (shift_labels != -100)
                if valid_mask.any():
                    loss = (losses * valid_mask).sum(1) / valid_mask.sum(1)
                else:
                    loss = torch.tensor(0.0, dtype=shift_logits.dtype, device=shift_logits.device)
                # (B,)
                loss = loss * gating_prob_k.reshape_as(loss)
                # (1,)
                loss = loss.mean()
            
        return loss, logits, outputs, gating_prob



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastWithGatingProb]:

        if self.training and self.get_model().is_m3:
            # "The model is in training mode."
            
            # map 'ver=v0_numtoks=[1,9,36,144,576]' to:
            # ['ver=v0_numtoks=1', 'ver=v0_numtoks=9', 'ver=v0_numtoks=36', 'ver=v0_numtoks=144', 'ver=v0_numtoks=576']
            matryoshka_vis_token_scale = self.get_model().config.config['matryoshka_vis_token_scale']
            kvs = parse_kv_from_string(matryoshka_vis_token_scale)
            if kvs['ver'] == 'v0':
                num_toks = eval(kvs['numtoks']) # str -> List
                matryoshka_vis_token_scale = []
                for num_tok in num_toks:
                    kvs['numtoks'] = str(num_tok)
                    matryoshka_vis_token_scale.append( create_string_from_kv(kvs) )
            else:
                raise ValueError(f"[llava.model.language_model.llava_llama.py] {kvs['ver']} not implemented.")

            loss = 0
            logits_accumulate = []
            for matryoshka_vis_token_scale_element in matryoshka_vis_token_scale:
                loss_item, logits, outputs, gating_prob = self.forward_single_matryoshka(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    past_key_values = past_key_values,
                    inputs_embeds = inputs_embeds,
                    labels = labels,
                    use_cache = use_cache,
                    output_attentions = output_attentions,
                    output_hidden_states = output_hidden_states,
                    images = images,
                    image_sizes = image_sizes,
                    return_dict = return_dict,
                    matryoshka_vis_token_scale = matryoshka_vis_token_scale_element,
                )
                if gating_prob is None:
                    loss += loss_item/len(matryoshka_vis_token_scale)
                else:
                    loss += loss_item
                logits_accumulate.append(logits)
                # wpq: only logits & loss is the avg of the different scales.
                #      `past_key_values`, `hidden_states`, `attentions` are from last scale only.
                #      this is ok since this conditional block is used in training only that just needs `loss`.
                #                 1 scale              multiple scales
                # loss:   (1,)                     ->  (1,)             
                # logits: (B, seq_len, vocab_size) ->  (B, seq_len_1+...+seq_len_K, vocab_size) where K=#token scales
                assert len(outputs) == 1, 'len(outputs) == 1 is False'
            logits = torch.cat(logits_accumulate, dim = 1)
                    
            if not return_dict:
                output = (logits,) + outputs[1:] + (gating_prob,)
                return (loss,) + output if loss is not None else output
            return CausalLMOutputWithPastWithGatingProb(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                gating_prob=gating_prob,
            )
            
        else:
            # "The model is in evaluation mode or trained without matryoshka_vis_token_scale."
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    gating_prob,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            if not return_dict:
                output = (logits,) + outputs[1:] + (gating_prob,)
                return (loss,) + output if loss is not None else output
            return CausalLMOutputWithPastWithGatingProb(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                gating_prob=gating_prob,
            )



    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        matryoshka_vis_token_scale = kwargs.pop("matryoshka_vis_token_scale", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                gating_prob,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                matryoshka_vis_token_scale = matryoshka_vis_token_scale
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
