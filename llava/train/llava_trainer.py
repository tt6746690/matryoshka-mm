from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from typing import List, Optional

import torch.distributed as dist
import wandb

try:
    from rosemary import parse_kv_from_string
except:
    pass


### wpq: imports for exposing `training_steps`
from packaging import version
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward
else:
    IS_SAGEMAKER_MP_POST_1_10 = False
if is_apex_available():
    from apex import amp
### 


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult = 8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper['mm_projector'] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper['vision_tower'] = self.args.mm_vision_tower_lr
            if self.args.router_lr is not None:
                lr_mapper['router'] = self.args.router_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": lr,
                        },
                    ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        # inputs = self._prepare_inputs(inputs)

        # print({
        #     'rank': self.args.process_index,
        #     'seq_len': (inputs['labels'] != -100).sum(1),
        # })

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # wpq: to get `ModelOutputs` instead of tuple.
        inputs.update({'return_dict': True})

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        ## wpq: log per-expert LM loss
        if outputs.losses is not None: # m3 type training
            if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                log_dict = {}
                losses_lm_reduced = torch.mean(outputs.losses_lm, 0) # (1,) unweighted lm loss
                for k in range(outputs.losses.numel()):
                    log_dict.update({f'moe/loss_lm_{k}': losses_lm_reduced[k].item()})


        ### wpq: MoE load balancing loss.
        if self.model.get_model().is_m3_moe:
            # use `self.model` to access llama model.
            kvs = parse_kv_from_string(self.model.config.config.get('moe', None))
            tokscales = eval(parse_kv_from_string(self.model.config.config.get('matryoshka_vis_token_scale', None)).get('numtoks', None))

            # (micro-bsz, K)
            gating_prob = outputs.gating_prob
            device, dtype = gating_prob.device, gating_prob.dtype
            assert(gating_prob.shape[1] == len(tokscales))

            with torch.no_grad():
                # gather `gating_prob`` (micro-bsz, K) -> (B, K) where K is number of experts.
                gating_prob_list = [torch.zeros_like(outputs.gating_prob) for _ in range(self.args.world_size)]
                dist.all_gather(gating_prob_list, outputs.gating_prob)
                batch_gating_prob = torch.cat(gating_prob_list, dim=0)
                B, K = batch_gating_prob.shape
                # (K,)
                batch_per_expert_gating_prob = torch.mean(gating_prob, dim=0)
                # (K,)
                batch_per_expert_assignment = torch.nn.functional.one_hot(batch_gating_prob.argmax(dim=1), num_classes=K)
                batch_per_expert_assignment = batch_per_expert_assignment.sum(dim=0).float() / B
                batch_per_expert_assignment = batch_per_expert_assignment.to(device).to(dtype)

            if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                for k in range(K):
                    log_dict.update({f'moe/avg_gating_prob_{k}': batch_per_expert_gating_prob[k].item()})
                for k in range(K):
                    log_dict.update({f'moe/avg_expert_assignment_{k}': batch_per_expert_assignment[k].item()})


            moe_objective_type = kvs.get('obj', 'weightedlm')
            if moe_objective_type.startswith('bounderr'):
                margin = float(kvs.get('margin', 0))
                # (micro-bsz, K)
                gating_prob_argmax = compute_gating_prob_argmax(gating_prob, kvs)
                # assume token scale sorted, largeest token scale at the end.
                # (micro-bsz, K)
                losses_lm = outputs.losses_lm
                # (micro-bsz)
                losses_argmaxscale = (losses_lm * gating_prob_argmax).sum(1)
                losses_maxtokscale = losses_lm[:, -1]
                losses_diff = losses_argmaxscale - losses_maxtokscale
                if moe_objective_type == 'bounderr':
                    loss = torch.clamp(losses_diff - margin, min=0).mean()
                elif moe_objective_type == 'bounderrsq':
                    loss = torch.square(torch.clamp(losses_diff - margin, min=0)).mean()

                if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                    log_dict.update({
                        'moe_bounderr/loss_argmaxscale_avg': losses_argmaxscale.mean().item(),
                        'moe_bounderr/loss_maxscale_avg': losses_maxtokscale.mean().item(),
                        'moe_bounderr/loss_diff_avg': losses_diff.mean().item(),
                    })
            elif moe_objective_type == 'weightedlm':
                pass

            ## compute switch transformer load balance loss: https://dl.acm.org/doi/pdf/10.5555/3586589.3586709
            if kvs.get('loadb', None) == 'switch':
                alpha = float(kvs['alpha'])
                per_expert_cost_type = kvs.get('costt', 'count')
                per_expert_cost = get_per_expert_cost(per_expert_cost_type, batch_per_expert_assignment, tokscales, device, dtype)
                # (K,), (K,) -> (,)
                loss_switch = alpha * K * (per_expert_cost * torch.mean(gating_prob, dim=0)).sum()
                loss += loss_switch
                
                if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                    log_dict.update({'moe_load/loss_switch': loss_switch.item(),})
                    for k in range(K):
                        log_dict.update({f'moe_load/cost_{k}': per_expert_cost[k].item()})
            elif kvs.get('loadb', None) == 'argmaxcost':
                # apply expert specific cost to argmax of `gating_prob`
                alpha = float(kvs['alpha'])
                per_expert_cost_type = kvs.get('costt')
                # since `argmaxcost` normalized to [0,1], therefore, select target value within [0,1] suffices.
                target_value = kvs.get('tval', None)
                numtoks_margin = kvs.get('tmargin', None)
                # (K,)
                per_expert_cost = get_per_expert_cost(per_expert_cost_type, batch_per_expert_assignment, tokscales, device, dtype)
                if not moe_objective_type.startswith('bounderr'): # already initialized `gating_prob_argmax`
                    # (micro-bsz, K)
                    gating_prob_argmax = compute_gating_prob_argmax(gating_prob, kvs)
                # (1,) micro-batch cost
                # since cost sums to 1, therefore sum wrt expert dimension
                argmaxcost = (gating_prob_argmax * per_expert_cost.reshape(-1, K)).sum(1).mean()
                # (1,) batch average cost
                # if just use micro-batch cost in loss, too noisy since micro-bsz is quite small (e.g., 4)
                with torch.no_grad():
                    argmaxcost_list = [torch.zeros_like(argmaxcost.unsqueeze(0)) for _ in range(self.args.world_size)]
                    dist.all_gather(argmaxcost_list, argmaxcost.unsqueeze(0))
                    argmaxcost_list = torch.cat(argmaxcost_list, dim=0)
                    batch_argmaxcost = argmaxcost_list.mean()
                if target_value is not None:
                    loss_argmaxcost = alpha * torch.square(argmaxcost - target_value)
                else:
                    loss_argmaxcost = alpha * torch.clamp(batch_argmaxcost - argmaxcost.detach() + argmaxcost - numtoks_margin, min=0)
                loss += loss_argmaxcost
                if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                    log_dict.update({'moe_load/loss_argmaxcost': loss_argmaxcost.item(),})
                    for k in range(K):
                        log_dict.update({f'moe_load/cost_{k}': per_expert_cost[k].item()})
            elif kvs.get('loadb', None) == 'betalogprob':
                if K != 2:
                    raise ValueError(f'#tokscale = {K} not supported.')
                alpha = float(kvs['alpha'])
                beta_alpha = float(kvs['ba'])
                beta_beta = float(kvs['bb'])
                beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
                log_prob = beta_dist.log_prob(gating_prob[:,1])
                loss_beta_logprob = alpha * log_prob.sum()
                loss += loss_beta_logprob
                if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                    log_dict.update({'moe_load/loss_beta_logprob': loss_beta_logprob.item(),})


         # log once/batch if assume no gradient accumulation.
        if self.is_world_process_zero() and 'wandb' in self.args.report_to:
            wandb.log(log_dict)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    


def compute_gating_prob_argmax(gating_prob, kvs):
    # if hard=True, taking argmax and therefore `tau` does not really matter.
    tau = float(kvs.get('tau', 1))
    hard = bool(kvs.get('hard', True))
    # (micro-bsz, K)
    gating_prob_argmax = torch.nn.functional.gumbel_softmax(gating_prob, tau=tau, hard=hard, dim=1)
    return gating_prob_argmax


def get_per_expert_cost(per_expert_cost_type, batch_per_expert_assignment, tokscales, device, dtype):
    if per_expert_cost_type == 'count': # default used in switch transformers
        per_expert_cost = batch_per_expert_assignment
    elif per_expert_cost_type == 'numtoks':
        per_expert_cost = torch.tensor(tokscales, device=device, dtype=dtype)
        per_expert_cost = per_expert_cost / per_expert_cost.sum()
    elif per_expert_cost_type == 'lognumtoks':
        per_expert_cost = torch.tensor(tokscales, device=device, dtype=dtype)
        per_expert_cost = torch.log(per_expert_cost+1) # add 1 to prevent cost(tokscale=1)=0
        per_expert_cost = per_expert_cost / per_expert_cost.sum()
    elif per_expert_cost_type == 'count*numtoks':
        per_expert_cost = batch_per_expert_assignment
        per_expert_cost_2 = torch.tensor(tokscales, device=device, dtype=dtype)
        per_expert_cost_2 = per_expert_cost_2 / per_expert_cost_2.sum()
        per_expert_cost *= per_expert_cost_2
        per_expert_cost = per_expert_cost / per_expert_cost.sum()
    elif per_expert_cost_type == 'count*lognumtoks':
        per_expert_cost = batch_per_expert_assignment
        per_expert_cost_2 = torch.tensor(tokscales, device=device, dtype=dtype)
        per_expert_cost_2 = torch.log(per_expert_cost_2+1) # add 1 to prevent cost(tokscale=1)=0
        per_expert_cost_2 = per_expert_cost_2 / per_expert_cost_2.sum()
        per_expert_cost *= per_expert_cost_2
        per_expert_cost = per_expert_cost / per_expert_cost.sum()
    else:
        raise ValueError(f'per_expert_cost_type={per_expert_cost_type} not supported.')
    return per_expert_cost