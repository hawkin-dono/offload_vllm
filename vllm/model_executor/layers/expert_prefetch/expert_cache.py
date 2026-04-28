import torch
import torch.nn as nn
from collections.abc import Callable
import threading
import queue

from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.fused_moe import FusedMoE


class PinnedBuffer:
    def __init__(self, num_experts, cached_parameter_names, sample_layer):
        self.params = {}
        self.ready_event = threading.Event()
        self.ready_event.set()
        for param_name in cached_parameter_names:
            source_param = getattr(sample_layer, param_name)
            self.params[param_name] = torch.empty(
                (num_experts, *source_param.shape[1:]),
                dtype=source_param.dtype,
                device="cpu",
                pin_memory=True
            )

class PinnedBufferPool:
    def __init__(self, pool_size, num_experts, cached_parameter_names, sample_layer):
        self.pool_size = pool_size
        self.buffers = [PinnedBuffer(num_experts, cached_parameter_names, sample_layer) 
                        for _ in range(pool_size)]
        
    def get_buffer(self, layer_id):
        return self.buffers[layer_id % self.pool_size]


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

    def fetch_on_demand(self, layer, expert_ids, slot_ids: torch.Tensor | None = None, layer_id: int | None = None):
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
        
        pinned_buffer = None
        if layer_id is not None and hasattr(layer.expert_cache, "pinned_pool") and getattr(layer.expert_cache, "pinned_pool") is not None:
            pinned_buffer = layer.expert_cache.pinned_pool.get_buffer(layer_id)
            pinned_buffer.ready_event.wait()
            
        for slot_id, expert_id in zip(slot_ids.tolist(), local_ids.tolist()):
            for param_name in cached_parameter_names:
                cache_param = getattr(self, param_name)
                
                if pinned_buffer is not None:
                    cpu_pinned_param = pinned_buffer.params[param_name]
                    cache_param[slot_id].copy_(cpu_pinned_param[slot_id], non_blocking=True)
                else:
                    # Fallback just in case
                    layer_param = getattr(layer, param_name)
                    cache_param[slot_id].copy_(
                        layer_param[expert_id].pin_memory(),
                        non_blocking=True,
                    )


class ExpertCache(nn.Module):
    def __init__(self, num_experts, predict_distance: int = 1):
        super().__init__()
        self.ping_buffer = ExpertBuffer(num_experts=num_experts, name="ping")
        self.pong_buffer = ExpertBuffer(num_experts=num_experts, name="pong")
        self.active_buffer = "ping"
        # Bound at model init so we can reuse FusedMoE's loader logic.
        self.owner_fused_moe: FusedMoE | None = None
        self.cached_parameter_names: tuple[str, ...] = ()
        
        self.predict_distance = predict_distance
        self.pinned_pool = None
        self.copy_queue = queue.Queue()
        self.copy_thread = threading.Thread(target=self._background_copy_loop, daemon=True)
        self.copy_thread.start()

    def _background_copy_loop(self):
        while True:
            task = self.copy_queue.get()
            if task is None:
                break
                
            target_layer, layer_id, expert_ids, slot_ids = task
            if self.pinned_pool is None:
                self.copy_queue.task_done()
                continue
                
            pinned_buffer = self.pinned_pool.get_buffer(layer_id)
            pinned_buffer.ready_event.clear()
            
            for slot_id, expert_id in zip(slot_ids, expert_ids):
                for param_name in self.cached_parameter_names:
                    layer_param = getattr(target_layer, param_name)
                    pinned_param = pinned_buffer.params[param_name]
                    pinned_param[slot_id].copy_(layer_param[expert_id])
                    
            pinned_buffer.ready_event.set()
            self.copy_queue.task_done()

    def add_to_copy_queue(self, target_layer, layer_id, expert_ids, slot_ids=None):
        if slot_ids is None:
            slot_ids = torch.arange(len(expert_ids), dtype=torch.long).tolist()
        if isinstance(expert_ids, torch.Tensor):
            expert_ids = expert_ids.tolist()
            
        self.copy_queue.put((target_layer, layer_id, expert_ids, slot_ids))

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
        self.owner_fused_moe = owner_fused_moe
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
            
        if getattr(self, "pinned_pool", None) is None:
            self.pinned_pool = PinnedBufferPool(
                pool_size=self.predict_distance + 1,
                num_experts=self.ping_buffer.num_experts,
                cached_parameter_names=self.cached_parameter_names,
                sample_layer=owner_fused_moe
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
        # OLD(ducct): explicit MXFP4-only registration.
        # self.register_parameter("expert_cache_ping_w13_weight", self.ping_buffer.w13_weight)
        # set_weight_attrs(self.ping_buffer.w13_weight, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w13_weight_scale", self.ping_buffer.w13_weight_scale)
        # set_weight_attrs(self.ping_buffer.w13_weight_scale, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w13_bias", self.ping_buffer.w13_bias)
        # set_weight_attrs(self.ping_buffer.w13_bias, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w2_weight", self.ping_buffer.w2_weight)
        # set_weight_attrs(self.ping_buffer.w2_weight, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w2_weight_scale", self.ping_buffer.w2_weight_scale)
        # set_weight_attrs(self.ping_buffer.w2_weight_scale, extra_weight_attrs)
        # self.register_parameter("expert_cache_ping_w2_bias", self.ping_buffer.w2_bias)
        # set_weight_attrs(self.ping_buffer.w2_bias, extra_weight_attrs)
        for param_name in self.cached_parameter_names:
            param = getattr(self.ping_buffer, param_name)
            self.register_parameter(f"expert_cache_ping_{param_name}", param)
            set_weight_attrs(param, extra_weight_attrs)

        self.pong_buffer.create_buffer(
            owner_fused_moe,
            self.cached_parameter_names,
        )
        # OLD(ducct): explicit MXFP4-only registration.
        # self.register_parameter("expert_cache_pong_w13_weight", self.pong_buffer.w13_weight)
        # set_weight_attrs(self.pong_buffer.w13_weight, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w13_weight_scale", self.pong_buffer.w13_weight_scale)
        # set_weight_attrs(self.pong_buffer.w13_weight_scale, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w13_bias", self.pong_buffer.w13_bias)
        # set_weight_attrs(self.pong_buffer.w13_bias, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w2_weight", self.pong_buffer.w2_weight)
        # set_weight_attrs(self.pong_buffer.w2_weight, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w2_weight_scale", self.pong_buffer.w2_weight_scale)
        # set_weight_attrs(self.pong_buffer.w2_weight_scale, extra_weight_attrs)
        # self.register_parameter("expert_cache_pong_w2_bias", self.pong_buffer.w2_bias)
        # set_weight_attrs(self.pong_buffer.w2_bias, extra_weight_attrs)
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
        predicted_expert_ids,
        prefetch_fn: Callable[[], None] | None = None,
        stream: torch.cuda.Stream | None = None,
    ):
        # NOTE(ducct): mark inactive buffer unavailable and record completion event.
        inactive_cache = self.get_inactive_buffer()
        inactive_cache.avail = False
        if torch.cuda.is_available():
            if stream is None:
                stream = torch.cuda.current_stream()
            with torch.cuda.stream(stream):
                if prefetch_fn is not None:
                    prefetch_fn()
                inactive_cache.prefetch_event = torch.cuda.Event()
                inactive_cache.prefetch_event.record(stream)
        else:
            if prefetch_fn is not None:
                prefetch_fn()
            inactive_cache.avail = True
