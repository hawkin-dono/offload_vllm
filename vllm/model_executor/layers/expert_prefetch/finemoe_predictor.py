
import torch 
import numpy as np 
from dataclasses import dataclass
import numpy as np
import hashlib
import networkx as nx


@dataclass
class ExpertTraceEntry:
    seq_id: str = None
    matrix: np.ndarray = None
    iters: dict = None
    num_new_tokens: int = 0
    num_prefill_tokens: int = 0

    def __hash__(self):
        return hash(self.seq_id)
    

class ExpertMapStore():
    def __init__(
        self,
        capacity,
        num_layers,
        num_experts,
        embed_dim,
        prefetch_distance,
        device,
    ):
        self.capacity = capacity
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.prefetch_distance = prefetch_distance
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.store_embed = torch.zeros(
            (capacity, embed_dim), dtype=self.dtype, device=self.device)
        self.store_traj = torch.zeros(
            (capacity, num_layers, num_experts), dtype=self.dtype, device=self.device)

        self.data_size = 0

    def import_store_data(self, state_path):
        self.store_embed = torch.from_numpy(
            np.load(f"{state_path}~embed~{self.capacity}.npy",
                    allow_pickle=False)
        ).to(self.device, dtype=self.dtype, non_blocking=True)

        self.store_traj = torch.from_numpy(
            np.load(f"{state_path}~traj~{self.capacity}.npy",
                    allow_pickle=False)
        ).to(self.device, dtype=self.dtype, non_blocking=True)

        self.data_size = self.store_embed.shape[0]

    def export_store_data(self, state_path):
        np.save(f"{state_path}~embed~{self.capacity}.npy",
                self.store_embed.detach().cpu().numpy(), allow_pickle=False)
        np.save(f"{state_path}~traj~{self.capacity}.npy",
                self.store_traj.detach().cpu().numpy(), allow_pickle=False)
        print(f"Data saved at {state_path}~embed~{self.capacity}.npy and {state_path}~traj~{self.capacity}.npy")

    @torch.inference_mode()
    def _cosine_sim(self, A: torch.Tensor, B: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        A = torch.nn.functional.normalize(A, dim=-1, eps=eps)
        B = torch.nn.functional.normalize(B, dim=-1, eps=eps)
        return A @ B.T

    def _ensure_tensor(self, x, shape_last=None):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)
        if shape_last is not None:
            assert t.shape[-len(shape_last):] == tuple(
                shape_last), f"Expected trailing shape {shape_last}, got {tuple(t.shape)}"
        return t.to(self.device, dtype=self.dtype, non_blocking=True)

    @torch.inference_mode()
    def add(self, embeds, expert_maps):
        embeds = self._ensure_tensor(embeds, shape_last=(self.embed_dim,))
        expert_maps = self._ensure_tensor(
            expert_maps, shape_last=(self.num_layers, self.num_experts))
        B = embeds.size(0)
        if B == 0:
            return

        free = self.capacity - self.data_size
        take = min(free, B)
        if take > 0:
            idx = slice(self.data_size, self.data_size + take)
            self.store_embed[idx] = embeds[:take]
            self.store_traj[idx] = expert_maps[:take]
            self.data_size += take

        rem = B - take
        if rem > 0:
            S_e = F.cosine_similarity(
                embeds[-rem:].unsqueeze(1),
                self.store_embed[:self.data_size].unsqueeze(0),
                dim=-1,
            )
            sims_t = F.cosine_similarity(
                expert_maps[-rem:].reshape(rem, -1).unsqueeze(1),
                self.store_traj[:self.data_size].reshape(
                    self.data_size, -1).unsqueeze(0),
                dim=-1,
            )
            S_t = sims_t
            w = self.prefetch_distance / float(self.num_layers)
            redundant = w * S_e + (1.0 - w) * S_t
            evict_idx = torch.argmax(redundant, dim=1)
            self.store_embed[evict_idx] = embeds[-rem:]
            self.store_traj[evict_idx] = expert_maps[-rem:]

        self.data_size = min(self.capacity, self.data_size)

    @torch.inference_mode()
    def match_embed(self, embeds):
        if self.data_size == 0:
            return None, None
        embeds = self._ensure_tensor(embeds, shape_last=(self.embed_dim,))
        sims = F.cosine_similarity(
            embeds.unsqueeze(1),
            self.store_embed[:self.data_size].unsqueeze(0),
            dim=-1,
        )
        scores, argmax = sims.max(dim=1)
        maps = self.store_traj[:self.data_size][argmax]
        return scores, maps

    @torch.inference_mode()
    def match_traj(self, trajs):
        if self.data_size == 0:
            return None, None
        trajs = self._ensure_tensor(trajs, shape_last=(
            trajs.shape[-2], self.num_experts))
        L_obs = trajs.shape[1]
        B = trajs.shape[0]
        sims = F.cosine_similarity(
            trajs[:, None, :L_obs, :].reshape(B, 1, -1),
            self.store_traj[:self.data_size, :L_obs, :][None,
                                                        :, :, :].reshape(1, self.data_size, -1),
            dim=-1,
        )
        scores, argmax = sims.max(dim=1)
        maps = self.store_traj[:self.data_size][argmax]
        return scores, maps


class ExpertTracer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExpertTracer, cls).__new__(cls)
        return cls._instance

    def __init__(self, capacity: int, config: PretrainedConfig, expert_map_store, eval_mode, device):
        self.num_layers, self.num_experts, self.num_encoder_layers, self.embed_dim, self.top_k = parse_moe_param(
            config)
        self.capacity = capacity

        self.expert_map_store = expert_map_store
        self.eval_mode = eval_mode
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.trace = {}

        self.trace_collection = torch.zeros(
            (capacity, self.num_layers, self.num_experts),
            dtype=self.dtype, device=self.device
        )
        self.collection_access = torch.zeros(
            (capacity,), dtype=torch.int32, device="cpu")

    def create_entry(self):
        seq_id = uuid.uuid4().hex
        self.trace[seq_id] = ExpertTraceEntry(
            seq_id=seq_id,
            matrix=torch.zeros((self.num_layers, self.num_experts),
                               dtype=self.dtype, device=self.device),
            iters=[],
            num_new_tokens=0,
            num_prefill_tokens=0,
        )
        return seq_id

    @torch.inference_mode()
    def finish_entry(self, seq_id):
        trace_sum = self.trace_collection.abs().sum(dim=(1, 2))
        empty_mask = trace_sum == 0
        if torch.any(empty_mask):
            idx = int(torch.nonzero(empty_mask, as_tuple=False)[0].item())
            self.trace_collection[idx] = self.trace[seq_id].matrix
            self.collection_access[idx] = 1
        else:
            idx = int(torch.argmin(self.collection_access).item())
            self.trace_collection[idx] = self.trace[seq_id].matrix
            self.collection_access[idx] = self.collection_access[idx] + 1

    @torch.inference_mode()
    def update_embed(self, seq_id: int, embeds: torch.Tensor):
        trace_entry = self.trace[seq_id]
        if len(trace_entry.iters) == 0:
            for i in range(embeds.shape[0]):
                trace_entry.iters.append({
                    "stage": "prefill",
                    "embed": embeds[i].detach(),
                    "states": torch.zeros((self.num_layers, self.embed_dim),  dtype=self.dtype, device=self.device),
                    "nodes":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                    "probs":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                    "preds":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                })
        else:
            assert trace_entry.iters[-1]["embed"] is not None, "Something is wrong with decode steps!"
            trace_entry.iters.append({
                "stage": "decode",
                "embed": embeds.detach().squeeze(0),
                "states": torch.zeros((self.num_layers, self.embed_dim),  dtype=self.dtype, device=self.device),
                "nodes":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                "probs":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                "preds":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
            })

    @torch.inference_mode()
    def _row_add_bincount(self, row_1d: torch.Tensor, idxs_2d: torch.Tensor):
        device = row_1d.device
        E = row_1d.numel()
        flat = idxs_2d.to(device=device, dtype=torch.long,
                          non_blocking=True).reshape(-1)
        inc = torch.bincount(flat, minlength=E).to(
            dtype=row_1d.dtype, device=device)
        row_1d.add_(inc)

    @torch.inference_mode()
    def _row_add_index_add(self, row_1d: torch.Tensor, idxs_1d: torch.Tensor):
        device = row_1d.device
        idxs_1d = idxs_1d.to(
            device=device, dtype=torch.long, non_blocking=True)
        ones = torch.ones_like(idxs_1d, dtype=row_1d.dtype, device=device)
        row_1d.index_add_(0, idxs_1d, ones)

    @torch.inference_mode()
    def update_entry(self, seq_id, expert_list: torch.Tensor, layer_idx: int,
                     hidden_states: torch.Tensor, expert_probs: torch.Tensor):
        expert_list = torch.as_tensor(expert_list, dtype=torch.long)
        expert_list = expert_list.to(self.device, non_blocking=True)

        hidden_states = torch.as_tensor(hidden_states, dtype=self.dtype)
        hidden_states = hidden_states.to(self.device, non_blocking=True)

        expert_probs = torch.as_tensor(expert_probs, dtype=self.dtype)
        expert_probs = expert_probs.to(self.device, non_blocking=True)

        trace_entry = self.trace[seq_id]
        num_tokens, _ = expert_list.shape

        completed_embeds = []
        completed_maps = []

        row_global = trace_entry.matrix[layer_idx]
        if num_tokens > 1:
            self._row_add_bincount(row_global, expert_list)
        else:
            self._row_add_index_add(row_global, expert_list[0])

        if num_tokens > 1:
            if trace_entry.num_prefill_tokens == 0:
                trace_entry.num_prefill_tokens = num_tokens

            for token_idx in range(num_tokens):
                it = trace_entry.iters[token_idx]
                states = it["states"]
                nodes = it["nodes"]
                probs = it["probs"]

                idxs = expert_list[token_idx]
                self._row_add_index_add(nodes[layer_idx], idxs)
                probs[layer_idx].copy_(expert_probs[token_idx])
                states[layer_idx].copy_(hidden_states[token_idx])

                if layer_idx == self.num_layers - 1:
                    completed_embeds.append(it["embed"])
                    completed_maps.append(probs)
        else:
            assert layer_idx < self.num_layers
            token_idx = trace_entry.num_prefill_tokens + trace_entry.num_new_tokens - 1
            it = trace_entry.iters[token_idx]
            states = it["states"]
            nodes = it["nodes"]
            probs = it["probs"]

            idxs = expert_list[0]
            self._row_add_index_add(nodes[layer_idx], idxs)
            probs[layer_idx].copy_(expert_probs[0])
            states[layer_idx].copy_(hidden_states[0])

            if layer_idx == self.num_layers - 1:
                completed_embeds.append(it["embed"])
                completed_maps.append(probs)

        if layer_idx == self.num_layers - 1:
            trace_entry.num_new_tokens += 1

        if self.eval_mode == "online" and completed_embeds and completed_maps:
            embeds_tensor = torch.stack(completed_embeds, dim=0)
            maps_tensor = torch.stack(completed_maps, dim=0)
            self.expert_map_store.add(
                embeds=embeds_tensor, expert_maps=maps_tensor)

    @torch.inference_mode()
    def update_preds(self, seq_id: int, iter_id: int, expert_preds: torch.Tensor, layer_start: int, layer_end: int):
        expert_preds = torch.as_tensor(
            expert_preds, dtype=self.dtype, device=self.device)
        trace_entry = self.trace[seq_id]
        it = trace_entry.iters[iter_id]
        preds = it["preds"]

        for layer_idx in range(layer_start, layer_end):
            assert layer_idx < self.num_layers, f"Invalid layer_idx {layer_idx}"
            layer_mask = expert_preds[layer_idx] != 0
            preds[layer_idx].zero_()
            preds[layer_idx][layer_mask] = 1

    def get_entry(self, seq_id):
        return self.trace[seq_id]

class ExpertMapMatcher():
    def __init__(
        self,
        expert_tracer,
        expert_map_store,
        expert_prefetcher,
        prefetch_distance,
    ):
        self.expert_tracer = expert_tracer
        self.expert_map_store = expert_map_store
        self.expert_prefetcher = expert_prefetcher

        self.prefetch_distance = prefetch_distance
        self.num_layers = self.expert_map_store.num_layers
        self.num_experts = self.expert_map_store.num_experts
        self.embed_dim = self.expert_map_store.embed_dim
        self.top_k = self.expert_tracer.top_k

        self.device = self.expert_map_store.device

    @torch.inference_mode()
    def _select_by_cumsum(self, probs: torch.Tensor, threshold: torch.Tensor, top_k: int):
        vals, idx = torch.sort(probs, dim=-1, descending=True)
        csum = vals.cumsum(-1)
        th = threshold.view(-1, 1) if threshold.ndim == 1 else threshold
        has_pos = vals.gt(0).any(-1)
        k = (csum <= th).sum(-1)
        k_nonzero = torch.clamp(k, min=top_k, max=vals.size(-1))
        k = torch.where(has_pos, k_nonzero, torch.zeros_like(k))
        ar = torch.arange(vals.size(-1), device=probs.device).unsqueeze(0)
        keep_sorted = ar < k.unsqueeze(-1)
        mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, src=keep_sorted)
        return probs * mask

    @torch.inference_mode()
    def _layer_decay_weights(self, layer_start: int, layer_end: int) -> torch.Tensor:
        assert layer_end > layer_start
        w = torch.ones(self.num_layers, device=self.device)
        rng = torch.arange(self.num_layers, device=self.device)
        band = (rng >= layer_start) & (rng < layer_end)
        w[band] = -1 / (layer_end + 1) * (rng[band] - layer_start) + 1
        return w

    @torch.inference_mode()
    def process_expert_map(self, layer_start: int, layer_end: int,
                           score: torch.Tensor, expert_map: torch.Tensor):
        probs = expert_map.clone()
        expert_prob_map = probs
        if layer_start > 0:
            probs[:layer_start, :].zero_()
        if layer_end < self.num_layers:
            probs[layer_end:, :].zero_()
        prefetch_priority_map = self._select_by_cumsum(
            probs, torch.clamp(1 - score, 0, 1), self.top_k)
        decay = self._layer_decay_weights(
            layer_start, layer_end).unsqueeze(-1)
        prefetch_priority_map = prefetch_priority_map * decay
        prefetch_priority_map[layer_start:layer_end] += 1e-6
        return prefetch_priority_map, expert_prob_map

    @torch.inference_mode()
    def embed_prefetch(self, seq_id: int, input_embeds: torch.Tensor):
        seq_len = input_embeds.shape[0]
        scores, maps = self.expert_map_store.match_embed(input_embeds)
        if scores is not None and maps is not None:
            layer_start = 0
            layer_end = self.prefetch_distance

            for i, (s, m) in enumerate(zip(scores, maps)):
                pred_map, prob_map = self.process_expert_map(
                    layer_start, layer_end, s, m)
                iter_id = i if seq_len > 1 else -1
                self.expert_tracer.update_preds(
                    seq_id, iter_id, pred_map, layer_start, layer_end)
                self.expert_prefetcher.prefetch_experts(
                    pred_map, prob_map)

    @torch.inference_mode()
    def traj_prefetch(self, seq_id: int, input_trajs: torch.Tensor):
        seq_len = input_trajs.shape[0]
        num_layers_obs = input_trajs.shape[1]
        layer_start = num_layers_obs + self.prefetch_distance
        if layer_start < self.num_layers:
            layer_end = self.num_layers
            scores, maps = self.expert_map_store.match_traj(
                input_trajs)

            if scores is not None and maps is not None:
                for i, (s, m) in enumerate(zip(scores, maps)):
                    pred_map, prob_map = self.process_expert_map(
                        layer_start, layer_end, s, m)
                    iter_id = i if seq_len > 1 else -1
                    self.expert_tracer.update_preds(
                        seq_id, iter_id, pred_map, layer_start, layer_end)
                    self.expert_prefetcher.prefetch_experts(
                        pred_map, prob_map)

