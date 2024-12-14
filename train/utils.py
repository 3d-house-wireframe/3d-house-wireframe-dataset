from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType
from beartype import beartype
from beartype.typing import Tuple
from math import ceil
from train.helpers import l1norm
from einops import rearrange, repeat, reduce, einsum


def save_static_dict_keys(static_dict, file_path='static_dict_keys.json'):
    # Flatten nested keys
    checkpoint_keys = extract_keys(static_dict)

    # Save keys to a text file
    with open(file_path, 'w') as f:
        for key in checkpoint_keys:
            f.write(f"{key}\n")

def extract_keys(d, parent_key=''):
    keys_list = []
    for k, v in d.items():
        # Build new key path
        new_key = f"{parent_key}/{k}" if parent_key else k
        # If the value is a dictionary, recursively call the function
        if isinstance(v, dict):
            keys_list.extend(extract_keys(v, new_key))
        else:
            keys_list.append(new_key)
    return keys_list

def load_model(model, load_path, device='cpu'):
    model_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = model_ckpt['ema_model']
    # Create a new state dictionary, removing the 'model.' prefix
    new_model_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('online_model.'):
            new_key = key.replace('online_model.', '')
            new_model_state_dict[new_key] = value

    load_status = model.load_state_dict(new_model_state_dict)
    # print(device)
    print(f"Autoencoder loaded from {load_path}")
    
    return model


def segment_coords_to_wireframe(segment_coords):
    # Rearrange line coordinates to form vertices
    vertices = rearrange(segment_coords, 'nl nlv c -> (nl nlv) c')
    lines = [[i, i + 1] for i in range(0, vertices.size(0), 2)]

    return dict(
        vertices=vertices,
        lines=lines
    )
    

def gen_to_wf(segment_batch_list, temp='0.0', batch_idx=0, method='3dwire'):

    temp = temp.replace('.', '_')

    dir_path = f'generate_lines/npz/' + method + '/temp_' + temp + '/'
    Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    for i, line_list in enumerate(segment_batch_list):

        points = line_list['vertices'].detach().cpu().numpy()

        lines = np.asarray(line_list['lines'])

        file_name = dir_path + f'/{str(batch_idx).zfill(2)}_{temp}_{str(i).zfill(3)}'
        
        np.savez(file_name, vertices=points, lines=lines)


@torch.no_grad()
def derive_positive_direction(vectors, eps=1e-8):
    mask_1 = vectors[:,:,0] < -eps
    vectors[mask_1] *= -1

    mask_2 = (vectors[:,:,0].abs() < eps) & (vectors[:,:,1] < -eps)
    vectors[mask_2] *= -1

    mask_3 = (vectors[:,:,0].abs() < eps) & (vectors[:,:,1].abs() < eps) & (vectors[:,:,2] < -eps)
    vectors[mask_3] *= -1

    return vectors

@torch.no_grad()
def get_derived_segment_features(
    segment_coords: TensorType['b', 'nl', 2, 3, float], # 2 vertices per segment, 3 coordinates per vertex
):
    vectors = segment_coords[:, :, 1, :] - segment_coords[:, :, 0, :]
    midps = (segment_coords[:, :, 1, :] + segment_coords[:, :, 0, :]) / 2
    length = vectors.norm(dim = -1, keepdim=True)    
    vectors_norm = vectors / (length + 1e-8)
    directions = derive_positive_direction(vectors_norm)
    
    return dict(
        length = length,
        directions = directions,
        midps = midps,
    )

@torch.no_grad()
def get_derived_segment_edge_features(
    segment_edge_coords: TensorType['b', 'e', 'nel', 'nlv', 'c', float], # 2 segments per segment_edge, 2 vertices per segment, 3 coordinates per vertice
    segment_edge_mask: TensorType['b', 'e', bool]
):
    segment_edge_vertex_0 = segment_edge_coords[:, :, :, 0, :]
    segment_edge_vertex_1 = segment_edge_coords[:, :, :, 1, :]
    
    segment_vectors = segment_edge_vertex_0 - segment_edge_vertex_1
    
    segment_edge_segment_norm = segment_vectors.norm(dim = -1)
    
    dot_product_all = einsum('ijk, ijk->ij', segment_vectors[:, :, 0, :], segment_vectors[:, :, 1, :])

    angle_radians = torch.acos(dot_product_all/(segment_edge_segment_norm[:, :, 0] * segment_edge_segment_norm[:, :, 1] + 1e-8))

    angle_radians = torch.where(angle_radians > torch.pi/2, 
                                torch.pi - angle_radians, 
                                angle_radians)

    angle_radians = angle_radians.masked_fill(~segment_edge_mask, float(0.))
    
    return angle_radians

@torch.no_grad()
def get_derived_segment_features_prob(
    segment_coords: TensorType['b', 'nd', 'nc', float], # num_coords(`nc`), num_discrete(`nd`) per coordinate
):
    segment_coords = rearrange(segment_coords, 'b nd nc -> b nl 2 3 nd')
    pass

# tensor helper functions

@beartype
def discretize(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)

@beartype
def undiscretize(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

@beartype
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.
) -> Tensor:

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2 
    half_width = width // 2 

    distance = torch.arange(-half_width, half_width + 1, dtype = dtype, device = device) 

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian) 

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels) 

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding = half_width, groups = channels) 
    return rearrange(out, 'b c n -> b n c')

@beartype
def scatter_mean(
    tgt: Tensor,
    indices: Tensor,
    src = Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-5
):
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min = eps)

@torch.no_grad()
def derive_segment_edges_from_segments(
    segments: TensorType['b', 'nl', 2, int],
    pad_id = -1,
    neighbor_if_share_one_vertex = True,
    include_self = True
) -> TensorType['b', 'e', 2, int]:

    is_one_segment, device = segments.ndim == 2, segments.device

    if is_one_segment:
        # add batch dimension
        segments = rearrange(segments, 'nl c -> 1 nl c')

    max_num_segments = segments.shape[1]
    segment_edges_vertices_threshold = 1 if neighbor_if_share_one_vertex else 2

    all_edges = torch.stack(torch.meshgrid(
        torch.arange(max_num_segments, device = device),
        torch.arange(max_num_segments, device = device),
    indexing = 'ij'), dim = -1)

    segment_masks = reduce(segments != pad_id, 'b nl c -> b nl', 'all') 
    segment_edges_masks = rearrange(segment_masks, 'b i -> b i 1') & rearrange(segment_masks, 'b j -> b 1 j')

    segment_edges = []

    for segment, segment_edge_mask in zip(segments, segment_edges_masks):

        shared_vertices = rearrange(segment, 'i c -> i 1 c 1') == rearrange(segment, 'j c -> 1 j 1 c')
        num_shared_vertices = shared_vertices.any(dim = -1).sum(dim = -1)
        is_neighbor_segment = (num_shared_vertices == segment_edges_vertices_threshold) & segment_edge_mask

        if not include_self:
            is_neighbor_segment &= num_shared_vertices != 2

        segment_edge = all_edges[is_neighbor_segment]
        segment_edge = torch.sort(segment_edge, dim=1).values
        
        # Use unique to find the unique rows in the sorted tensor
        # Since the pairs are sorted, [1, 0] and [0, 1] will both appear as [0, 1] and be considered the same
        segment_edge, _ = torch.unique(segment_edge, sorted=True, return_inverse=True, dim=0)

        segment_edges.append(segment_edge)

    segment_edges = pad_sequence(segment_edges, padding_value = pad_id, batch_first = True)

    if is_one_segment:
        segment_edges = rearrange(segment_edges, '1 e ij -> e ij')

    # get segment_edge features
    
    return segment_edges
