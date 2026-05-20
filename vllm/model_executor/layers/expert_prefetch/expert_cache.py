import torch
import torch.nn as nn
from collections.abc import Callable

from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.fused_moe import FusedMoE
from concurrent.futures import ThreadPoolExecutor
import threading

class ExpertBuffer(nn.Module):
    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w13_bias: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    w2_bias: torch.Tensor

    def __init__(self, num_experts: int, name: str):
        super().__init__()
        self.num_experts = num_experts
        # Keep cached expert IDs on GPU when available to avoid CPU<->GPU hops.
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.cached_expert_ids: torch.Tensor = torch.empty(
            0, dtype=torch.int32, device=device
        )
        self.avail = True
        self.prefetch_event: torch.cuda.Event | None = None
        self.cpu_done_event = threading.Event()
        self.cpu_done_event.set() # Mặc định ban đầu là xong

    def record_expert_ids(self, expert_ids: torch.Tensor):
        self.cached_expert_ids = expert_ids

    def get_cached_expert_ids(self):
        return self.cached_expert_ids

    def is_avail(self):
        return self.avail

    def create_buffer(
        self,
        layer: FusedMoE,
        cached_parameter_names: tuple[str, ...],
    ):
        

        for param_name in cached_parameter_names:
            source_param = getattr(layer, param_name)
            cached_param = torch.nn.Parameter(
                torch.zeros(
                    (self.num_experts, *source_param.shape[1:]),
                    dtype=source_param.dtype,
                    # device="cuda",
                ),
                requires_grad=False,
            )
            print(f"cached_param: {cached_param.device}")
            setattr(self, param_name, cached_param)

    def _copy_float8_rows(self, dst, src, expert_ids, n):
        # dst: GPU float8 tensor, src: CPU float8 tensor
        dst_u8 = dst[:n].view(torch.uint8)
        src_u8 = src.view(torch.uint8)[expert_ids]
        dst_u8.copy_(src_u8)

    def fetch_on_demand(self, layer, expert_ids, slot_ids: torch.Tensor | None = None):
        expert_ids = expert_ids.reshape(-1)
        if expert_ids.numel() == 0:
            return

        # Map global expert ids -> local expert ids when EP is enabled.
        # Filter out experts not owned by this rank (mapped to -1).
        # with torch.profiler.record_function("expert_ids.map_and_filter"):
        if getattr(layer, "expert_map", None) is not None:  #not matter, since run on 1 device
            map_device = layer.expert_map.device
            local_ids = layer.expert_map[
                expert_ids.to(map_device, dtype=torch.long)
            ]
            keep = local_ids >= 0
            if not torch.any(keep):
                return
            local_ids = local_ids[keep]
        else:
            local_ids = expert_ids

        # with torch.profiler.record_function("expert_ids.to_device"):
        # local_ids = local_ids.to(layer.w13_weight.device, dtype=torch.long)
        if slot_ids is not None:
            slot_ids = slot_ids.to(layer.w13_weight.device, dtype=torch.long)

        num_expert_ids = local_ids.numel()

        if slot_ids is None:
            slot_ids = torch.arange(
                num_expert_ids,
                device=layer.w13_weight.device,
                dtype=torch.long,
            )

        cached_parameter_names = getattr(
            layer.expert_cache,
            "cached_parameter_names",
            (),
        )
        for slot_id, expert_id in zip(slot_ids.tolist(), local_ids.tolist()):
            for param_name in cached_parameter_names:
                cache_param = getattr(self, param_name)
                layer_param = getattr(layer, param_name)
                # print(f"cache_param: {cache_param.device}")
                # print(f"layer_param: {layer_param.device}")
                cache_param[slot_id].copy_(
                    layer_param[expert_id].pin_memory(),
                    non_blocking=True,
                )
                
    def chunking_prefetch(self, layer, expert_ids, slot_ids: torch.Tensor | None = None, chunk_size: int = 4):
        expert_ids = expert_ids.reshape(-1)
        if expert_ids.numel() == 0:
            return

        if getattr(layer, "expert_map", None) is not None:  #not matter, since run on 1 device
            map_device = layer.expert_map.device
            local_ids = layer.expert_map[
                expert_ids.to(map_device, dtype=torch.long)
            ]
            keep = local_ids >= 0
            if not torch.any(keep):
                return
            local_ids = local_ids[keep]
        else:
            local_ids = expert_ids

        if slot_ids is not None:
            slot_ids = slot_ids.to(layer.w13_weight.device, dtype=torch.long)

        num_expert_ids = local_ids.numel()

        if slot_ids is None:
            slot_ids = torch.arange(
                num_expert_ids,
                device=layer.w13_weight.device,
                dtype=torch.long,
            )

        cached_parameter_names = getattr(
            layer.expert_cache,
            "cached_parameter_names",
            (),
        )
        
        slot_list = slot_ids.tolist()
        expert_list = local_ids.tolist()
        total_experts = len(slot_list)
        for start_idx in range(0, total_experts, chunk_size):
            end_idx = min(start_idx + chunk_size, total_experts)
            chunk_slots = slot_list[start_idx:end_idx]
            chunk_experts = expert_list[start_idx:end_idx]

            for slot_id, expert_id in zip(chunk_slots, chunk_experts):
                for param_name in cached_parameter_names:
                    cache_param = getattr(self, param_name)
                    layer_param = getattr(layer, param_name)
                    cache_param[slot_id].copy_(
                        layer_param[expert_id].pin_memory(),
                        non_blocking=True,
                    )
                
            torch.cuda.current_stream().synchronize()


class ExpertCache(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.ping_buffer = ExpertBuffer(num_experts=num_experts, name="ping")
        self.pong_buffer = ExpertBuffer(num_experts=num_experts, name="pong")
        self.active_buffer = "ping"
        # Bound at model init so we can reuse FusedMoE's loader logic.
        self.owner_fused_moe: FusedMoE | None = None
        self.cached_parameter_names: tuple[str, ...] = ()
        self.prefetch_executor = ThreadPoolExecutor(max_workers=1)

    # NOTE(ducct): custom weight loader for expert cache
    def cached_weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        selected_expert_ids: list[int],
        key: str,
        return_success: bool = False,
    ) -> bool | None:
        owner = self.owner_fused_moe
        if owner is None:
            raise AttributeError(
                "ExpertCache.owner_fused_moe is not set; cannot reuse "
                "FusedMoE.cached_weight_loader."
            )
        return FusedMoE.cached_weight_loader(
            owner,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            selected_expert_ids=selected_expert_ids,
            key=key,
            return_success=return_success,
        )


    def _resolve_cached_parameter_names(self, layer: FusedMoE) -> tuple[str, ...]:
        quant_method_name = layer.quant_method.__class__.__name__
        print(f"quant_method_name: {quant_method_name}")
        if quant_method_name == "Mxfp4MoEMethod":
            return (
                "w13_weight",
                "w13_weight_scale",
                "w13_bias",
                "w2_weight",
                "w2_weight_scale",
                "w2_bias",
            )
        if quant_method_name == "UnquantizedFusedMoEMethod":
            print(f"[ducct] I'm in here brooo")
            names = ["w13_weight", "w2_weight"]
            if hasattr(layer, "w13_bias"):
                names.append("w13_bias")
            if hasattr(layer, "w2_bias"):
                names.append("w2_bias")
            return tuple(names)

        return tuple(
            name
            for name in (
                "w13_weight",
                "w13_weight_scale",
                "w13_bias",
                "w2_weight",
                "w2_weight_scale",
                "w2_bias",
            )
            if hasattr(layer, name)
        )

    def create_cache(self, owner_fused_moe: FusedMoE):
        # OLD(ducct): fixed MXFP4-only cache init signature.
        # def create_cache(
        #     self,
        #     config,
        #     mxfp4_block,
        #     weight_dtype,
        #     scale_dtype,
        # ):
        # self.owner_fused_moe = owner_fused_moe
        self.__dict__["owner_fused_moe"] = owner_fused_moe
        extra_weight_attrs = {
            "cached_weight_loader": self.cached_weight_loader
        }
        self.cached_parameter_names = self._resolve_cached_parameter_names(
            owner_fused_moe
        )
        if not self.cached_parameter_names:
            raise NotImplementedError(
                "Expert cache could not determine cached parameter layout for "
                f"{owner_fused_moe.quant_method.__class__.__name__}."
            )
        # Avoid reallocating shared buffers if they already exist.
        if (
            self.ping_buffer is not None
            and self.pong_buffer is not None
            and getattr(self.ping_buffer, "w13_weight", None) is not None
            and getattr(self.pong_buffer, "w13_weight", None) is not None
        ):
            return
        self.ping_buffer.create_buffer(
            owner_fused_moe,
            self.cached_parameter_names,
        )

        for param_name in self.cached_parameter_names:
            param = getattr(self.ping_buffer, param_name)
            self.register_parameter(f"expert_cache_ping_{param_name}", param)
            set_weight_attrs(param, extra_weight_attrs)

        self.pong_buffer.create_buffer(
            owner_fused_moe,
            self.cached_parameter_names,
        )

        for param_name in self.cached_parameter_names:
            param = getattr(self.pong_buffer, param_name)
            self.register_parameter(f"expert_cache_pong_{param_name}", param)
            set_weight_attrs(param, extra_weight_attrs)

    def get_active_buffer(self):
        if self.active_buffer == "ping":
            return self.ping_buffer
        return self.pong_buffer

    def get_inactive_buffer(self):
        if self.active_buffer == "ping":
            return self.pong_buffer
        return self.ping_buffer

    def flip_active_buffer(self):
        self.active_buffer = "pong" if self.active_buffer == "ping" else "ping"

    def prefetch(
        self,
        layer: FusedMoE,
        p_ids: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
        inactive_buffer: ExpertBuffer | None = None,
    ):
        inactive_buffer.avail = False
        
        inactive_buffer.cpu_done_event.clear()
        target_device = inactive_buffer.w13_weight.device
        self.prefetch_executor.submit(
            self.back_ground_prefetch, layer, inactive_buffer, p_ids, stream, target_device)
            
    def back_ground_prefetch(
        self,
        layer: FusedMoE,
        inactive_buffer: ExpertBuffer,
        p_ids: torch.Tensor,
        stream: torch.cuda.Stream | None,
        target_device: torch.device,
    ):
        try:
            torch.cuda.set_device(target_device)
            if torch.cuda.is_available():
                if stream is None:
                    stream = torch.cuda.current_stream()
                with torch.cuda.stream(stream):
                    inactive_buffer.chunking_prefetch(layer, p_ids, chunk_size=self.owner_fused_moe.top_k //2) 
                    inactive_buffer.prefetch_event = torch.cuda.Event()
                    inactive_buffer.prefetch_event.record(stream)
            else:
                inactive_buffer.avail = True
        finally:
            inactive_buffer.cpu_done_event.set()