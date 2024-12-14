from math import ceil
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from x_transformers import Decoder
from x_transformers.autoregressive_wrapper import top_k

from tqdm import tqdm
from .segment_vqvae import SegmentVQVAE
from .modules import GateLoopBlock

from train.helpers import (
    pad_to_length, divisible_by,
    safe_cat, exists, default,
    set_module_requires_grad_
)


class WireframeTransformer(Module):
    def __init__(
        self,
        autoencoder: SegmentVQVAE,
        dim = 512,
        max_seq_len = 8192,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs = dict(
            ff_glu = True,
            attn_num_mem_kv = 4
        ),
        dropout = 0.,
        coarse_pre_gateloop_depth = 2,
        fine_pre_gateloop_depth = 2,
        gateloop_use_heinsen = False,
        fine_attn_depth = 2,
        pad_id = -1,
    ):
        super().__init__()

        dim, dim_fine = (dim, dim) if isinstance(dim, int) else dim

        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        self.sos_token = nn.Parameter(torch.randn(dim_fine))
        self.eos_token_id = self.codebook_size

        assert divisible_by(max_seq_len, 2 * self.num_quantizers), f'max_seq_len ({max_seq_len}) must be divisible by (2 x {self.num_quantizers}) = {2 * self.num_quantizers}' # 2 vertices per line, with D codes per vertex

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)

        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.vertex_embed = nn.Parameter(torch.randn(2, dim))
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        self.to_line_tokens = nn.Sequential(
            nn.Linear(self.num_quantizers * 2 * dim, dim),
            nn.LayerNorm(dim)
        )

        self.coarse_gateloop_block = GateLoopBlock(dim, depth = coarse_pre_gateloop_depth, use_heinsen=gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else None

        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            attn_dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            **attn_kwargs
        )

        self.maybe_project_coarse_to_fine = nn.Linear(dim, dim_fine) if dim != dim_fine else nn.Identity()

        self.fine_gateloop_block = GateLoopBlock(dim, depth = fine_pre_gateloop_depth, use_heinsen=gateloop_use_heinsen) if fine_pre_gateloop_depth > 0 else None

        self.fine_decoder = Decoder(
            dim = dim_fine,
            depth = fine_attn_depth,
            attn_dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            **attn_kwargs
        )

        self.to_logits = nn.Linear(dim_fine, self.codebook_size + 1)

        self.pad_id = pad_id
        autoencoder.pad_id = pad_id

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(
        self,
        prompt = None,
        batch_size = None,
        filter_logits_fn = top_k,
        filter_kwargs = dict(),
        temperature = 1.,
        return_codes = False,
        cond_scale = 1.,
        cache_kv = True,
        max_seq_len = None,
        segment_coords_to_file = None
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        batch_size = default(batch_size, 1)

        codes = default(prompt, torch.empty((batch_size, 0), dtype = torch.long, device = self.device))

        curr_length = codes.shape[-1]

        cache = (None, None)

        for i in tqdm(range(curr_length, max_seq_len)):
            code_len = codes.shape[-1]

            can_eos = i != 0 and divisible_by(i, self.num_quantizers * 2) 

            output = self.forward_on_codes(
                codes,
                return_loss = False,
                return_cache = cache_kv,
                append_eos = False,
            )

            if cache_kv:
                logits, cache = output

                if cond_scale == 1.:
                    cache = (cache, None)
            else:
                logits = output

            logits = logits[:, -1]

            if (not can_eos):
                logits[:, -1] = -torch.finfo(logits.dtype).max

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)
            

            if temperature == 0.:
                sample = filtered_logits.argmax(dim = -1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim = -1) 
                sample = torch.multinomial(probs, 1)
            

            codes, _ = pack([codes, sample], 'b *')

            is_eos_codes = (codes == self.eos_token_id)

            if is_eos_codes.any(dim = -1).all():
                break

        mask = is_eos_codes.float().cumsum(dim = -1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)

        code_len = codes.shape[-1]
        round_down_code_len = code_len // self.num_quantizers * self.num_quantizers
        codes = codes[:, :round_down_code_len]

        code_len = codes.shape[-1]
        round_down_code_len = code_len // self.num_quantizers * self.num_quantizers
        codes = codes[:, :round_down_code_len]
    
        if return_codes:
            codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)
            return codes

        self.autoencoder.eval()
        
        segment_coords, segment_mask = self.autoencoder.codes_to_segments(codes)

        if not exists(segment_coords_to_file):
            return segment_coords, segment_mask

        files = [segment_coords_to_file(coords[mask]) for coords, mask in zip(segment_coords, segment_mask)]
        
        return files

    def forward(
        self,
        *,
        vertices = None,
        segments = None,
        segment_edges = None,
        codes = None,
        cache = None,
        **kwargs
    ):
        if not exists(codes):
            codes = self.autoencoder.segment_to_codes(
                vertices = vertices,
                segments = segments,
                segment_edges = segment_edges
            )
        
        return self.forward_on_codes(codes, cache = cache, **kwargs)

    def forward_on_codes(
        self,
        codes = None,
        return_loss = True,
        return_cache = False,
        append_eos = True,
        cache = None,
    ):

        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        batch, seq_len, device = *codes.shape, codes.device

        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        if append_eos:
            assert exists(codes)

            code_lens = ((codes == self.pad_id).cumsum(dim = -1) == 0).sum(dim = -1) 
            
            pad_length = 1
                
            codes = F.pad(codes, (0, pad_length), value = self.pad_id)
            
            batch_arange = torch.arange(batch, device = device)

            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')
            
            codes[batch_arange, code_lens] = self.eos_token_id

        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes 

        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        seq_arange = torch.arange(codes.shape[-2], device = device)

        codes = codes + self.abs_pos_emb(seq_arange)

        code_len = codes.shape[-2]
                
        second_part_len = codes.shape[-2]

        level_embed = repeat(self.quantize_level_embed, 'q d -> (r q) d', r = ceil(second_part_len / self.num_quantizers))
        codes = codes + level_embed[:second_part_len]

        vertex_embed = repeat(self.vertex_embed, 'nlv d -> (r nlv q) d', r = ceil(second_part_len / (2 * self.num_quantizers)), q = self.num_quantizers)
        codes = codes + vertex_embed[:second_part_len]
        
        num_tokens_per_segment = self.num_quantizers * 2 

        curr_vertex_pos = code_len % num_tokens_per_segment

        code_len_is_multiple_of_line = divisible_by(code_len, num_tokens_per_segment) 

        next_multiple_code_len = ceil(code_len / num_tokens_per_segment) * num_tokens_per_segment

        codes = pad_to_length(codes, next_multiple_code_len, dim = -2)        
        
        grouped_codes = rearrange(codes, 'b (nl n) d -> b nl n d', n = num_tokens_per_segment) 

        segment_codes = grouped_codes if code_len_is_multiple_of_line else grouped_codes[:, :-1] 

        segment_codes = rearrange(segment_codes, 'b nl n d -> b nl (n d)')
        segment_codes = self.to_line_tokens(segment_codes) 

        segment_codes_len = segment_codes.shape[-2]

        (
            cached_attended_segment_codes,
            coarse_cache,
            fine_cache,
            coarse_gateloop_cache,
            fine_gateloop_cache
        ) = cache if exists(cache) else ((None,) * 5)

        if exists(cache):
            cached_line_codes_len = cached_attended_segment_codes.shape[-2]
            need_call_first_transformer = segment_codes_len > cached_line_codes_len
        else:
            need_call_first_transformer = True

        should_cache_fine = not divisible_by(curr_vertex_pos + 1, num_tokens_per_segment)

        if need_call_first_transformer:
            if exists(self.coarse_gateloop_block):
                segment_codes, coarse_gateloop_cache = self.coarse_gateloop_block(segment_codes, cache = coarse_gateloop_cache)

            attended_segment_codes, coarse_cache = self.decoder(
                segment_codes,
                cache = coarse_cache,
                return_hiddens = True,
            )

            attended_segment_codes = safe_cat((cached_attended_segment_codes, attended_segment_codes), dim = -2)
        else:
            attended_segment_codes = cached_attended_segment_codes

        attended_segment_codes = self.maybe_project_coarse_to_fine(attended_segment_codes)

        sos = repeat(self.sos_token, 'd -> b d', b = batch)

        attended_line_codes_with_sos, _ = pack([sos, attended_segment_codes], 'b * d')
        
        grouped_codes = pad_to_length(grouped_codes, attended_line_codes_with_sos.shape[-2], dim = 1)
        fine_vertex_codes, _ = pack([attended_line_codes_with_sos, grouped_codes], 'b n * d')

        fine_vertex_codes = fine_vertex_codes[..., :-1, :]

        if exists(self.fine_gateloop_block):
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b nl n d -> b (nl n) d')
            orig_length = fine_vertex_codes.shape[-2]
            fine_vertex_codes = fine_vertex_codes[:, :(code_len + 1)]

            fine_vertex_codes, fine_gateloop_cache = self.fine_gateloop_block(fine_vertex_codes, cache = fine_gateloop_cache)

            fine_vertex_codes = pad_to_length(fine_vertex_codes, orig_length, dim = -2)
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b (nl n) d -> b nl n d', n = num_tokens_per_segment)

        if exists(cache):
            fine_vertex_codes = fine_vertex_codes[:, -1:]

            if exists(fine_cache):
                for attn_intermediate in fine_cache.attn_intermediates:
                    ck, cv = attn_intermediate.cached_kv
                    ck, cv = map(lambda t: rearrange(t, '(b nl) ... -> b nl ...', b = batch), (ck, cv))
                    ck, cv = map(lambda t: t[:, -1, :, :curr_vertex_pos], (ck, cv))
                    attn_intermediate.cached_kv = (ck, cv)

        one_line = fine_vertex_codes.shape[1] == 1

        fine_vertex_codes = rearrange(fine_vertex_codes, 'b nl n d -> (b nl) n d')

        if one_line:
            fine_vertex_codes = fine_vertex_codes[:, :(curr_vertex_pos + 1)]

        attended_vertex_codes, fine_cache = self.fine_decoder(
            fine_vertex_codes,
            cache = fine_cache,
            return_hiddens = True
        )

        if not should_cache_fine:
            fine_cache = None

        if not one_line:
            embed = rearrange(attended_vertex_codes, '(b nl) n d -> b (nl n) d', b = batch)
            embed = embed[:, :(code_len + 1)]
        else:
            embed = attended_vertex_codes
        
        if not return_loss:
            
            logits = self.to_logits(embed)
            
            if not return_cache:
                return logits

            next_cache = (
                attended_segment_codes,
                coarse_cache,
                fine_cache,
                coarse_gateloop_cache,
                fine_gateloop_cache
            )
            
            return logits, next_cache
        
        logits = self.to_logits(embed)
        
        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.pad_id
        )

        return ce_loss, None