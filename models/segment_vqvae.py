from math import pi
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from functools import partial
from einops import rearrange, repeat, reduce, pack
from einops.layers.torch import Rearrange
from x_transformers.x_transformers import RMSNorm, FeedForward
from local_attention import LocalMHA
from vector_quantize_pytorch import ResidualLFQ
from torch_geometric.nn.conv import SAGEConv
from .modules import ResnetBlock

from train.helpers import exists, default, is_odd, first
from train.utils import (
    discretize, 
    get_derived_segment_features,
    get_derived_segment_edge_features,
    scatter_mean,
    undiscretize,
    derive_segment_edges_from_segments,
    gaussian_blur_1d,
)


class SegmentVQVAE(Module):
    def __init__(
        self,
        num_discrete_coors = 128,
        coor_continuous_range = (-1., 1.),
        dim_coor_embed = 64,
        num_discrete_length = 128,
        dim_length_embed = 16,
        num_discrete_directions = 128,
        dim_direction_embed = 64,
        num_discrete_midps = 128,
        dim_midp_embed = 16,
        num_discrete_angle = 128,
        dim_angle_embed = 16,
        encoder_dims_through_depth = (
            64, 128, 256, 256, 384
        ),
        init_decoder_conv_kernel = 7,
        decoder_dims_through_depth = (
            128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256, 256, 256,
            384, 384, 384
        ),
        dim_codebook = 192,
        num_quantizers = 2,     
        codebook_size = 8192,     
        rlfq_kwargs = dict(),
        sageconv_kwargs = dict(
            normalize = True,
            project = True
        ),
        commit_loss_weight = 0.1,
        bin_smooth_blur_sigma = 0.4,  
        attn_encoder_depth = 0,
        attn_decoder_depth = 0,
        local_attn_kwargs = dict(
            dim_head = 32,
            heads = 8
        ),
        local_attn_window_size = 64,
        pad_id = -1,
        attn_dropout = 0.,
        ff_dropout = 0.,
        resnet_dropout = 0,
        checkpoint_quantizer = False,
    ):
        super().__init__()

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_segment_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        self.discretize_length = partial(discretize, num_discrete = num_discrete_length, continuous_range = (0., 2.))
        self.length_embed = nn.Embedding(num_discrete_length, dim_length_embed)

        self.discretize_midps = partial(discretize, num_discrete = num_discrete_midps, continuous_range = coor_continuous_range)
        self.midp_embed = nn.Embedding(num_discrete_midps, dim_midp_embed)

        self.discretize_directions = partial(discretize, num_discrete = num_discrete_directions, continuous_range = coor_continuous_range)
        self.direction_embed = nn.Embedding(num_discrete_directions, dim_direction_embed)

        self.discretize_angle = partial(discretize, num_discrete = num_discrete_angle, continuous_range = (0., pi/2.))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        attn_kwargs = dict(
            causal = False,
            prenorm = True,
            dropout = attn_dropout,
            window_size = local_attn_window_size,
        )

        init_dim = dim_coor_embed * 6 + dim_length_embed + dim_direction_embed * 3 + dim_angle_embed + dim_midp_embed * 3

        self.project_in = nn.Linear(init_dim, dim_codebook)

        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(dim_codebook, init_encoder_dim, **sageconv_kwargs)

        self.init_encoder_act_and_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(init_encoder_dim)
        )

        self.encoders = ModuleList([])

        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                curr_dim,
                dim_layer,
                **sageconv_kwargs
            )

            self.encoders.append(sage_conv)
            curr_dim = dim_layer
                
        self.encoder_attn_blocks = ModuleList([])

        for _ in range(attn_encoder_depth):
            self.encoder_attn_blocks.append(nn.ModuleList([
                LocalMHA(dim = curr_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(curr_dim), FeedForward(curr_dim, glu = True, dropout = ff_dropout))
            ]))

        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        
        self.project_dim_codebook = nn.Linear(curr_dim, dim_codebook * 2)

        self.quantizer = ResidualLFQ(
            dim = dim_codebook,
            num_quantizers = num_quantizers,
            codebook_size = codebook_size,
            commitment_loss_weight = 1.,
            **rlfq_kwargs,
        )

        self.checkpoint_quantizer = checkpoint_quantizer

        self.pad_id = pad_id

        decoder_input_dim = dim_codebook * 2

        self.decoder_attn_blocks = ModuleList([])

        for _ in range(attn_decoder_depth):
            self.decoder_attn_blocks.append(nn.ModuleList([
                LocalMHA(dim = decoder_input_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(decoder_input_dim), FeedForward(decoder_input_dim, glu = True, dropout = ff_dropout))
            ]))

        init_decoder_dim, *decoder_dims_through_depth = decoder_dims_through_depth
        curr_dim = init_decoder_dim

        assert is_odd(init_decoder_conv_kernel)

        self.init_decoder_conv = nn.Sequential(
            nn.Conv1d(dim_codebook * 2, init_decoder_dim, kernel_size = init_decoder_conv_kernel, padding = init_decoder_conv_kernel // 2),
            nn.SiLU(),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(init_decoder_dim),
            Rearrange('b n c -> b c n')
        )

        self.decoders = ModuleList([])

        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout = resnet_dropout)

            self.decoders.append(resnet_block)
            curr_dim = dim_layer

        self.to_coor_logits = nn.Sequential(
            nn.Linear(curr_dim, num_discrete_coors * 6),
            Rearrange('... (v c) -> ... v c', v = 6)
        )

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    def device(self):
        return next(self.parameters()).device

    def encode(
        self,
        *,
        vertices,
        segments,
        segment_edges,
        segment_mask,
        segment_edges_mask,
        return_segment_coordinates = False
    ):
        batch, _, num_coors, device = *vertices.shape, vertices.device
        _, num_segments, _ = segments.shape
        _, num_segment_edges, _ = segment_edges.shape

        segment_without_pad = segments.masked_fill(~rearrange(segment_mask, 'b nl -> b nl 1'), 0)
        segments_vertices = repeat(segment_without_pad, 'b nl nlv -> b nl nlv c', c = num_coors)
        vertices_for_segments = repeat(vertices, 'b nv c -> b nl nv c', nl = num_segments)

        segment_coords = vertices_for_segments.gather(-2, segments_vertices)        

        segment_edges_wo_pad = segment_edges.masked_fill(~rearrange(segment_edges_mask, 'b e -> b e 1'), 0)
        
        segment_edge_vertex = repeat(segment_edges_wo_pad, 'b e nel -> b e nel nlv c', nlv = 2, c = num_coors)
        vertices_for_segment_edges = repeat(segment_coords, 'b nl nlv c -> b e nl nlv c', e = num_segment_edges)

        segment_edge_coords = vertices_for_segment_edges.gather(2, segment_edge_vertex) 

        derived_segment_features = get_derived_segment_features(segment_coords)
        
        discrete_midps = self.discretize_midps(derived_segment_features['midps'])
        midp_embed = self.midp_embed(discrete_midps)

        discrete_length = self.discretize_length(derived_segment_features['length'])
        length_embed = self.length_embed(discrete_length)

        discrete_directions = self.discretize_directions(derived_segment_features['directions'])
        direction_embed = self.direction_embed(discrete_directions)

        derived_segment_edge_features = get_derived_segment_edge_features(segment_edge_coords, segment_edges_mask)
        
        discrete_angle = self.discretize_angle(derived_segment_edge_features)
        segment_edge_angle_embed = self.angle_embed(discrete_angle)
        
        dim_angle = segment_edge_angle_embed.shape[-1]
        
        segment_angle_embed = torch.zeros((batch, num_segments + 1, dim_angle), device = device)
        segment_edge_angle_embed = repeat(segment_edge_angle_embed, 'b e d -> b (e repeat) d', repeat=2)
        
        segment_edges_wo_pad = segment_edges_wo_pad.masked_fill(~rearrange(segment_edges_mask, 'b e -> b e 1'), num_segments)
        
        segment_edges_index = rearrange(segment_edges_wo_pad, 'b e nel -> b (e nel) 1')
        segment_edges_index = repeat(segment_edges_index, 'b enel 1 -> b enel repeat', repeat = dim_angle)
        
        averaged_segment_angle_embed = scatter_mean(segment_angle_embed, segment_edges_index, segment_edge_angle_embed, dim = -2)
        
        angle_embed = averaged_segment_angle_embed[:, :num_segments, :]

        discrete_segment_coords = self.discretize_segment_coords(segment_coords)
        discrete_segment_coords = rearrange(discrete_segment_coords, 'b nl nv c -> b nl (nv c)') 

        segment_coor_embed = self.coor_embed(discrete_segment_coords) 
        segment_coor_embed = rearrange(segment_coor_embed, 'b nl c d -> b nl (c d)') 

        segment_embed, _ = pack([segment_coor_embed, length_embed, direction_embed, angle_embed, midp_embed], 'b nl *')        
        segment_embed = self.project_in(segment_embed)

        segment_index_offsets = reduce(segment_mask.long(), 'b nl -> b', 'sum')
        segment_index_offsets = F.pad(segment_index_offsets.cumsum(dim = 0), (1, -1), value = 0)
        segment_index_offsets = rearrange(segment_index_offsets, 'b -> b 1 1')

        segment_edges = segment_edges + segment_index_offsets
        segment_edges = segment_edges[segment_edges_mask]
        segment_edges = rearrange(segment_edges, 'be ij -> ij be')

        orig_segment_embed_shape = segment_embed.shape[:2]

        segment_embed = segment_embed[segment_mask]

        segment_embed = self.init_sage_conv(segment_embed, segment_edges)

        segment_embed = self.init_encoder_act_and_norm(segment_embed)

        for conv in self.encoders:
            segment_embed = conv(segment_embed, segment_edges)

        shape = (*orig_segment_embed_shape, segment_embed.shape[-1])

        segment_embed = segment_embed.new_zeros(shape).masked_scatter(rearrange(segment_mask, '... -> ... 1'), segment_embed)

        for attn, ff in self.encoder_attn_blocks:
            segment_embed = attn(segment_embed, mask = segment_mask) + segment_embed
            segment_embed = ff(segment_embed) + segment_embed

        if not return_segment_coordinates:
            return segment_embed

        return segment_embed, discrete_segment_coords

    def quantize(
        self,
        *,
        segments,
        segment_mask,
        segment_embed,
        pad_id = None,
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, _, device = *segments.shape[:2], segments.device

        max_vertex_index = segments.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        segment_embed = self.project_dim_codebook(segment_embed)
        segment_embed = rearrange(segment_embed, 'b nl (nv d) -> b nl nv d', nv = 2)

        vertex_dim = segment_embed.shape[-1]
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        pad_vertex_id = num_vertices
        vertices = F.pad(vertices, (0, 0, 0, 1), value = 0.)

        segments = segments.masked_fill(~rearrange(segment_mask, 'b n -> b n 1'), pad_vertex_id)

        segments_with_dim = repeat(segments, 'b nl nv -> b (nl nv) d', d = vertex_dim)

        segment_embed = rearrange(segment_embed, 'b ... d -> b (...) d')

        averaged_vertices = scatter_mean(vertices, segments_with_dim, segment_embed, dim = -2)

        mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        mask[:, -1] = False 

        quantize_kwargs = dict(mask = mask)

        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant = False)

        quantized, codes, commit_loss = quantize_wrapper_fn((averaged_vertices, quantize_kwargs))

        segment_embed_output = quantized.gather(-2, segments_with_dim)
        segment_embed_output = rearrange(segment_embed_output, 'b (nl nlv) d -> b nl (nlv d)', nlv = 2)

        segments_with_quantized_dim = repeat(segments, 'b nl nv -> b (nl nv) q', q = self.num_quantizers)
        codes_output = codes.gather(-2, segments_with_quantized_dim)

        segment_mask = repeat(segment_mask, 'b nl -> b (nl nv) 1', nv = 2)
        codes_output = codes_output.masked_fill(~segment_mask, self.pad_id)

        return segment_embed_output, codes_output, commit_loss

    def decode(
        self,
        quantized,
        segment_mask
    ):
        conv_segment_mask = rearrange(segment_mask, 'b n -> b 1 n')

        x = quantized

        for attn, ff in self.decoder_attn_blocks:
            x = attn(x, mask = segment_mask) + x
            x = ff(x) + x

        x = rearrange(x, 'b n d -> b d n')

        x = x.masked_fill(~conv_segment_mask, 0.)
        x = self.init_decoder_conv(x)

        for resnet_block in self.decoders:
            x = resnet_block(x, mask = conv_segment_mask)

        return rearrange(x, 'b d n -> b n d')

    @torch.no_grad()
    def codes_to_segments(
        self,
        codes,
        segment_mask = None,
        return_discrete_codes = False
    ):
        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(segment_mask):
            segment_mask = reduce(codes != self.pad_id, 'b (nl nv q) -> b nl', 'all', nv = 2, q = self.num_quantizers)

        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, 'b (nl nv) d -> b nl (nv d)', nv = 2)

        decoded = self.decode(
            quantized,
            segment_mask=segment_mask
        )

        decoded = decoded.masked_fill(~segment_mask[..., None], 0.)
        pred_segment_coords = self.to_coor_logits(decoded)

        pred_segment_coords = pred_segment_coords.argmax(dim = -1)

        pred_segment_coords = rearrange(pred_segment_coords, '... (v c) -> ... v c', v = 2)

        continuous_coors = undiscretize(
            pred_segment_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        continuous_coors = continuous_coors.masked_fill(~rearrange(segment_mask, 'b nf -> b nf 1 1'), float('nan'))

        if not return_discrete_codes:
            return continuous_coors, segment_mask

        return continuous_coors, pred_segment_coords, segment_mask

    @torch.no_grad()
    def segment_to_codes(self, vertices, segments, segment_edges = None, *args, **kwargs):

        inputs = [vertices, segments, segment_edges]
        inputs = [*filter(exists, inputs)] 
        ndims = {i.ndim for i in inputs}

        assert len(ndims) == 1
        batch_less = first(list(ndims)) == 2

        if batch_less:
            inputs = [rearrange(i, '... -> 1 ...') for i in inputs]

        input_kwargs = dict(zip(['vertices', 'segments', 'segment_edges'], inputs))

        self.eval()

        codes = self.forward(
            **input_kwargs,
            return_codes = True,
            **kwargs
        )

        if batch_less:
            codes = rearrange(codes, '1 ... -> ...')

        return codes

    def forward(
        self,
        *,
        vertices,
        segments,
        segment_edges = None,
        return_codes = False,
        return_loss_breakdown = False,
        return_recon_segments = False,
        only_return_recon_segments = False,
        rvq_sample_codebook_temp = 1.
    ):
        if not exists(segment_edges):
            segment_edges = derive_segment_edges_from_segments(segments, pad_id = self.pad_id)

        segment_mask = reduce(segments != self.pad_id, 'b nl c -> b nl', 'all')
        segment_edges_mask = reduce(segment_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, segment_coordinates = self.encode(
            vertices = vertices,
            segments = segments,
            segment_edges = segment_edges,
            segment_edges_mask=segment_edges_mask,
            segment_mask = segment_mask,
            return_segment_coordinates = True
        )

        quantized, codes, commit_loss = self.quantize(
            segment_embed = encoded,
            segments = segments,
            segment_mask=segment_mask,
            rvq_sample_codebook_temp = rvq_sample_codebook_temp
        )

        if return_codes:
            assert not return_recon_segments, 'cannot return reconstructed segments when just returning raw codes'

            codes = codes.masked_fill(~repeat(segment_mask, 'b nf -> b (nf 2) 1'), self.pad_id)
            return codes

        decode = self.decode(
            quantized,
            segment_mask = segment_mask
        )

        pred_segment_coords = self.to_coor_logits(decode) 

        if return_recon_segments or only_return_recon_segments:

            recon_segments = undiscretize(
                pred_segment_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )

            recon_segments = rearrange(recon_segments, 'b nl (nv c) -> b nl nv c', nv = 2)
            segment_mask = rearrange(segment_mask, 'b nl -> b nl 1 1')
            recon_segments = recon_segments.masked_fill(~segment_mask, float('nan'))

        if only_return_recon_segments:
            return recon_segments

        pred_segment_coords = rearrange(pred_segment_coords, 'b ... c -> b c (...)')
        segment_coordinates = rearrange(segment_coordinates, 'b ... -> b 1 (...)')
        
        with autocast(enabled = False):
            pred_log_prob = pred_segment_coords.log_softmax(dim = 1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, segment_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma = self.bin_smooth_blur_sigma) 

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim = 1) 

            segment_mask = repeat(segment_mask, 'b nl -> b (nl r)', r = 6)
            recon_loss = recon_losses[segment_mask].mean()

        total_loss = recon_loss + commit_loss.sum() * self.commit_loss_weight

        loss_breakdown = {'recon_loss': recon_loss, 'commit_loss': commit_loss.sum()}

        if not return_loss_breakdown:
            if not return_recon_segments:
                return total_loss

            return recon_segments, total_loss

        if not return_recon_segments:
            return total_loss, loss_breakdown
        
        return recon_segments, total_loss, loss_breakdown

