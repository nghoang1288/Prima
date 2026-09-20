import logging
import os
import torch
import math
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from torch import nn
from perceiver_pytorch import Perceiver
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncoding1D
from einops.layers.torch import Rearrange
try:
    from torch.nn.attention.varlen import varlen_attn as torch_varlen_attn
except (ImportError, AttributeError):
    torch_varlen_attn = None

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_qkvpacked_func = None
# Avoid repeating the same backend line for every transformer layer.
_REPORTED_ATTENTION_BACKENDS = set()


# helpers


def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert input to tuple if it's not already."""
    return t if isinstance(t, tuple) else (t, t)


# wrapper for GPT, with MLP head
class GPTWrapper(nn.Module):
    """Wrapper for GPT model with MLP head for feature projection."""

    def __init__(self, model: nn.Module, feature_dim: int, model_dim: int):
        super().__init__()
        self.model = model
        self.mlp_head = nn.Sequential(nn.LayerNorm(model_dim),
                                      nn.Linear(model_dim, feature_dim),
                                      nn.LayerNorm(feature_dim))

    def forward(self, x: torch.Tensor, xmax: torch.Tensor) -> torch.Tensor:
        out = self.model(x)['last_hidden_state']
        bs, _, es = out.size()
        outs = torch.zeros(bs, es).to(out.device)
        for i in range(len(xmax)):
            outs[i] = out[i][xmax[i].item() - 1]
        return self.mlp_head(outs)


class PreNorm(nn.Module):
    """Pre-normalization wrapper for transformer blocks."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self,
                x: torch.Tensor,
                culen: Optional[torch.Tensor] = None,
                mxlen: Optional[torch.Tensor] = None):
        if culen is None:
            return self.fn(self.norm(x))
        return self.fn(self.norm(x), culen, mxlen)


class FeedForward(nn.Module):
    """Feed-forward network for transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,
                                           dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention with capability-based backend selection.

    Auto order: PyTorch native varlen, external flash-attn, then SDPA.
    Override with PRIMA_ATTENTION_BACKEND=auto|native|flash|sdpa.
    """

    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.,
                 causal: bool = False):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropoutp = dropout
        self.causal = causal
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def _backend(self) -> str:
        if hasattr(self, 'noflashattn') and self.noflashattn:
            return 'sdpa'
        requested = os.environ.get('PRIMA_ATTENTION_BACKEND', 'auto').lower()
        if requested not in {'auto', 'native', 'flash', 'sdpa'}:
            raise ValueError(
                'PRIMA_ATTENTION_BACKEND must be auto, native, flash, or sdpa'
            )
        if requested != 'auto':
            return requested
        if torch_varlen_attn is not None:
            return 'native'
        if flash_attn_varlen_qkvpacked_func is not None:
            return 'flash'
        return 'sdpa'

    def forward(self, x: torch.Tensor, culen: torch.Tensor,
                mxlen: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'causal'):
            self.causal = False

        bxs, _ = x.size()
        qkv = self.to_qkv(x).view(bxs, 3, self.heads, self.dim_head)
        cu = culen.to(device=x.device, dtype=torch.int32)
        max_len = int(mxlen.item()) if isinstance(mxlen, torch.Tensor) else int(mxlen)
        dropout_p = self.dropoutp if self.training else 0.0
        backend = self._backend()
        # Native varlen currently has no dropout_p argument. Preserve training
        # semantics by using flash-attn/SDPA when attention dropout is active.
        if backend == 'native' and self.training and dropout_p > 0:
            backend = 'flash' if flash_attn_varlen_qkvpacked_func is not None else 'sdpa'

        if backend == 'native':
            if torch_varlen_attn is None:
                if os.environ.get('PRIMA_ATTENTION_BACKEND', 'auto').lower() == 'native':
                    raise RuntimeError('Native PyTorch varlen attention is unavailable')
                backend = 'flash' if flash_attn_varlen_qkvpacked_func is not None else 'sdpa'
            else:
                q, k, v = qkv.unbind(dim=1)
                try:
                    out = torch_varlen_attn(
                        query=q,
                        key=k,
                        value=v,
                        cu_seq_q=cu,
                        cu_seq_k=cu,
                        max_q=max_len,
                        max_k=max_len,
                        scale=self.scale,
                        window_size=(-1, 0) if self.causal else (-1, -1),
                    )
                except torch.OutOfMemoryError:
                    # OOM is a capacity problem, not a backend capability miss.
                    # Retrying another attention implementation can worsen peak
                    # memory and obscure the actual 8 GB limit.
                    raise
                except (RuntimeError, NotImplementedError, ValueError, TypeError):
                    if os.environ.get('PRIMA_ATTENTION_BACKEND', 'auto').lower() == 'native':
                        raise
                    backend = 'flash' if flash_attn_varlen_qkvpacked_func is not None else 'sdpa'

        if backend == 'flash':
            if flash_attn_varlen_qkvpacked_func is None:
                if os.environ.get('PRIMA_ATTENTION_BACKEND', 'auto').lower() == 'flash':
                    raise RuntimeError('flash-attn backend requested but package is unavailable')
                backend = 'sdpa'
            else:
                out = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu,
                    max_len,
                    dropout_p=dropout_p,
                    causal=self.causal,
                )

        if backend == 'sdpa':
            out = sdpa_varlen(qkv, cu, dropout_p=dropout_p, causal=self.causal)

        is_compiling = (
            hasattr(torch, "compiler")
            and torch.compiler.is_compiling()
        )
        if not is_compiling:
            report_key = (backend, str(x.device), str(x.dtype))
            if report_key not in _REPORTED_ATTENTION_BACKENDS:
                _REPORTED_ATTENTION_BACKENDS.add(report_key)
                logging.info(
                    "PRIMA attention backend: %s device=%s dtype=%s",
                    backend,
                    x.device,
                    x.dtype,
                )

        out = out.flatten(start_dim=1)
        assert out.ndim == 2
        assert out.size(-1) == self.inner_dim
        return self.to_out(out)


def sdpa_varlen(
    qkv: torch.Tensor,
    culen: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    """Variable-length attention through PyTorch SDPA without batch padding."""
    q, k, v = qkv.unbind(dim=1)
    out = torch.empty_like(q)
    boundaries = culen.detach().to(device='cpu', dtype=torch.int64).tolist()

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        qs = q[start:stop].transpose(0, 1).unsqueeze(0)
        ks = k[start:stop].transpose(0, 1).unsqueeze(0)
        vs = v[start:stop].transpose(0, 1).unsqueeze(0)
        ys = F.scaled_dot_product_attention(
            qs, ks, vs, dropout_p=dropout_p, is_causal=causal
        )
        out[start:stop] = ys.squeeze(0).transpose(0, 1)
    return out


def no_flash_attn_varlen_substitute(qkv: torch.Tensor,
                                    culen: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for the historical non-flash fallback."""
    return sdpa_varlen(qkv, culen, dropout_p=0.0, causal=False)


class Transformer(nn.Module):
    """Transformer block with attention and feed-forward layers."""

    def __init__(self,
                 dim: int,
                 depth: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 dropout: float = 0.,
                 causal: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Attention(dim,
                                  heads=heads,
                                  dim_head=dim_head,
                                  dropout=dropout,
                                  causal=causal)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x: torch.Tensor, culen: torch.Tensor,
                mxlen: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, culen, mxlen) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """Vision Transformer for processing image sequences."""

    def __init__(self,
                 dim: int,
                 num_classes: int,
                 depth: int,
                 heads: int,
                 mlp_dim: int,
                 pool: str = 'cls',
                 dim_head: int = 64,
                 dropout: float = 0.,
                 emb_dropout: float = 0.,
                 clsnum: int = 10):
        """
        dim: dimension of embeddings in this transformer
        num_classes: final MLP head output dimension
        depth: depth of transforemr
        heads: attention heads of transformer
        mlp_dim: intermediate dimension of transformer MLP
        pool: either 'cls' (meaning classification token) or 'mean'
        dim_head: dimension of each attention head
        dropout: dropout rate
        emb_dropout: embedding dropout rate
        clsnum: how many classification tokens
        """
        super().__init__()
        self.dim = dim

        # num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)

        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pos_embedding = None
        self.clsnum = clsnum
        self.cls_token = nn.Parameter(torch.randn(1, self.clsnum, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.layernorm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.Linear(self.clsnum * dim,
                                                num_classes))

    # This function is used to ban flash attention from this ViT
    def make_no_flashattn(self) -> None:
        for layer in self.transformer.layers:
            layer[0].fn.noflashattn = True

    def forward(
        self,
        xdict: Dict[str, torch.Tensor],
        retpool: bool = False,
        retboth: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # xdict: the dictionary of inputs
        # retpool: whether to return the pooled ViT output without MLP
        # retboth: return both pooled ViT with and without MLP
        if not hasattr(self, 'dim'):
            self.dim = self.layernorm.normalized_shape[0]
        x = xdict['visual']
        lens = xdict['lens']  # the lengths of each sequence in the batch
        b, n, embsize = x.shape

        cls_tokens = repeat(self.cls_token, '1 c d -> b c d',
                            b=b)  # the classification tokens
        x = torch.cat((cls_tokens, x), dim=1)  # attach classification tokens
        # add positional embedding if using it
        if self.pos_embedding is not None:
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Pack variable-length sequences without Python-side tensor copies.
        # Row-major boolean indexing preserves the historical batch/token order.
        lengths = (lens + self.clsnum).to(device=x.device, dtype=torch.long)
        cl = torch.cumsum(lengths, dim=0)
        culen = torch.cat((cl.new_zeros(1), cl), dim=0)
        mxlen = lengths.max()
        token_positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        valid = token_positions < lengths.unsqueeze(1)
        x = x[valid]

        zerocheck = torch.logical_and(
            (x.max(dim=1).values == 0),
            (x.min(dim=1).values == 0)
        )
        assert zerocheck.int().sum() == 0
        xx = self.transformer(x, culen, mxlen)

        # Gather classification tokens from each packed sequence in one op.
        starts = culen[:-1].to(dtype=torch.long)
        cls_offsets = starts.unsqueeze(1) + torch.arange(
            self.clsnum, device=xx.device, dtype=torch.long
        ).unsqueeze(0)
        x = xx[cls_offsets]

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0:self.clsnum]

        x = self.to_latent(x)
        x = self.layernorm(x).view(b, -1)
        if retboth:
            return x, self.mlp_head(x)
        if retpool:
            return x
        return self.mlp_head(x)


# The LSTM that encodes the serienames
class SerieEncoder(nn.Module):
    """LSTM-based encoder for series names."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(47, 200)
        self.lstm = nn.LSTM(200, 200, batch_first=True)
        self.linear = nn.Linear(200, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x, _ = self.lstm(x)
        ret = self.linear(x[:, -1, :])
        return ret

    def make_no_flashattn(self) -> None:
        pass


# the transformer that encodes the serienames
class SerieTransformerEncoder(nn.Module):
    """Transformer-based encoder for series names."""

    def __init__(self, out_dim: int, positional_encoding_dim: int = 10):
        super().__init__()
        self.embsize = 256
        self.embed = nn.Embedding(47, 246)
        self.transformer = Transformer(dim=256,
                                       depth=3,
                                       heads=4,
                                       dim_head=64,
                                       mlp_dim=300)
        self.linear = nn.Linear(256, out_dim)
        p_enc = PositionalEncoding1D(positional_encoding_dim)
        self.p_enc = p_enc(torch.zeros(1, 200, positional_encoding_dim))[0]
        self.prelinear = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        occur46 = (x == 46).nonzero()
        assert occur46.size()[0] == len(x)
        assert occur46.size()[1] == 2
        assert torch.all(occur46[:, 0] == torch.arange(
            end=len(x)).to(occur46.device))
        lens = occur46[:, 1] + 1
        x = self.embed(x)
        posenc = self.p_enc[0:x.size()[1]].to(
            device=x.device, dtype=x.dtype
        ).unsqueeze(0).expand(len(x), -1, -1)
        x = torch.cat([x, posenc], dim=2)

        lengths = lens.to(device=x.device, dtype=torch.long)
        cl = torch.cumsum(lengths, dim=0)
        culen = torch.cat((cl.new_zeros(1), cl), dim=0)
        mxlen = lengths.max()
        token_positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        valid = token_positions < lengths.unsqueeze(1)
        nx = x[valid]
        nx = self.transformer(nx, culen, mxlen)
        xout = nx[(culen[1:] - 1).to(dtype=torch.long)]
        if hasattr(self, 'prelinear') and self.prelinear:
            return xout
        return self.linear(xout)

    # This function is used to ban flash attention from this module
    def make_no_flashattn(self) -> None:
        for layer in self.transformer.layers:
            layer[0].fn.noflashattn = True


# Hierarchial ViT
class HierViT(nn.Module):
    """Hierarchical Vision Transformer for processing medical image studies."""

    def __init__(self,
                 argsinner: Dict,
                 argsouter: Dict,
                 useseriename: bool = False,
                 usestudydescription: bool = False,
                 patdis: bool = False,
                 patdisdim: int = 128,
                 pretrainedserieencoder: Optional[str] = None):
        # argsinner: config for inner ViT (sequence ViT)
        # argsouter: config for outer ViT (study ViT)
        # useseriename: whether to use serie name (i.e. sequence names) for innerViT
        # usestudydescription: whether to use study description for outerViT
        # patdisdim: the dimension to project serie encoding to for patient discrimination loss
        # pretrainedserieencoder: a pre-trained serie name encoder. None if don't have one, will train one from random initialization.
        super().__init__()
        self.innerViT = ViT(**argsinner)  # the serie ViT
        self.outerViT = ViT(**argsouter)  # the study ViT
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.useseriename = useseriename
        self.usestudydescription = usestudydescription
        self.patdis = patdis
        self.patdisnet = torch.nn.Sequential(
            torch.nn.Linear(argsinner['clsnum'] * argsinner['dim'], 500),
            torch.nn.ReLU(), torch.nn.Linear(500, patdisdim))
        if useseriename:
            if pretrainedserieencoder is None:  # initialize new seriename encoder
                self.serieencoder = SerieTransformerEncoder(argsinner['dim'])
            else:  # use pre-trained seriename encoder, with linear layer mapping
                serieencoder = torch.load(pretrainedserieencoder,
                                          map_location='cpu').module.text_model
                serieencoder.prelinear = True
                self.serieencoder = torch.nn.Sequential(
                    serieencoder,
                    torch.nn.Linear(serieencoder.embsize, argsinner['dim']))
        if usestudydescription:
            self.studyencoder = SerieTransformerEncoder(argsouter['dim'])

    def forward(
        self,
        xdict: Dict[str, torch.Tensor],
        retpool: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # xdict: the dictionary of inputs
        #self.usestudydescription=False
        # retpool: whether to return the pooled ViT output without MLP
        mydevice = self.dummy_param.device
        lenss = xdict['lenss']  # lenss is length of each serie. Size is batch_size x max_series_per_study
        lens = xdict['lens']  # lens is num series per study
        x = xdict['visual']
        lenss = lenss.transpose(
            0, 1)  # Now it's max_series_per_study x batch size

        # how many extra spaces we need
        extra = 0
        if self.useseriename:
            extra += 1

        mapper = {
        }  # a dictionary mapping newx location to the serie's study number and serie number (i.e. the ith serie in the jth study)
        counter = 0  # counter tracks the location we're at in filling newx
        slens = []

        if not hasattr(self.innerViT,"dim"):
            self.innerViT.dim = 289

        # Batch-process series-name embeddings only when the checkpoint uses them.
        serienameencoded = None
        if self.useseriename:
            serienameencoded = self.dummy_param.new_zeros(
                (len(lens), len(lenss), self.innerViT.dim)
            )
            for j in range(len(lens)):
                serienamevecs = xdict['serienames'][j][0:lens[j]]
                serienameencoded[j][0:lens[j]] = self.serieencoder(serienamevecs)

        totalimgs = lens.sum(
        )  # totalimgs is total number of series over the batch
        imgmax = lenss.max()  # longest serie in batch
        newx = self.dummy_param.new_zeros(
            (int(totalimgs.item()), int(imgmax.item()) + extra, self.innerViT.dim)
        )  # the input to the innervit, where each series is treated as a separate sequence

        for i in range(len(x)):
            im = x[
                i]  # the collection of the ith series in all studies in batch

            for j in range(len(im)):
                if lenss[i][j] > 0:  # if the ith series of the jth study exist
                    mapper[counter] = (i, j)
                    slen = lenss[i][
                        j]  # length of the ith series of the jth study
                    if self.useseriename:
                        newx[counter][0] = serienameencoded[j][i]
                    theserie = im[j][0:slen]
                    theserielastdim = theserie.shape[-1]
                    newx[counter][
                        extra:slen + extra, 0:
                        theserielastdim] = theserie  # leave extra space for serie encoding
                    counter += 1
                    slens.append(slen + extra)
        slens = torch.stack(slens).to(
            device=mydevice, dtype=torch.long
        )

        innerraw, outs = self.innerViT({
            'visual': newx,
            'lens': slens
        },
                                       retboth=True)  # encode series

        extra = 0
        if self.usestudydescription:  # leave extra space for study description
            extra = 1
        nextx = outs.new_zeros(
            (len(lens), int(lens.max().item()) + extra, outs.size(-1))
        )  # the input to outervit, with size batch_size x max_num_series_per_study x emb_size
        if self.usestudydescription:
            nextx[:,
                  0] = self.studyencoder(xdict['studydescription'].long().to(
                      mydevice))  # include study description encoding
        for pos, out in enumerate(outs):
            i, j = mapper[pos]
            nextx[j][i + extra] = out
        outer_lens = lens + extra
        if self.patdis:
            m = []
            for pos in mapper:
                i, j = mapper[pos]
                m.append(hash(xdict['hash'][j]) % 1000000)
            assert len(m) == len(innerraw)
            m = torch.LongTensor(m).to(innerraw.device)
            return self.outerViT({
                'visual': nextx,
                'lens': outer_lens.to(mydevice)
            },
                                 retpool=retpool), self.patdisnet(innerraw), m
        if hasattr(self, 'getserieemb') and self.getserieemb:
            return self.outerViT({
                'visual': nextx,
                'lens': outer_lens.to(mydevice)
            },
                                 retpool=retpool), outs
        if hasattr(self, 'retboth') and self.retboth:
            return self.outerViT({
                'visual': nextx,
                'lens': outer_lens.to(mydevice)
            },
                                 retboth=True)
        return self.outerViT({
            'visual': nextx,
            'lens': outer_lens.to(mydevice)
        },
                             retpool=retpool)

    # This function is used to ban flash attention from this HierViT
    def make_no_flashattn(self) -> None:
        self.innerViT.make_no_flashattn()
        self.outerViT.make_no_flashattn()
        if hasattr(self, 'serieencoder'):
            try:
                self.serieencoder.make_no_flashattn()
            except (AttributeError, TypeError):
                self.serieencoder[0].make_no_flashattn()
        if hasattr(self, 'studyencoder'):
            self.studyencoder.make_no_flashattn()


# the clip objsctive
def clip_objective(
    d1: torch.Tensor,
    d2: torch.Tensor,
    temperature: torch.Tensor = torch.zeros(1)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute CLIP objective loss."""
    dots = torch.matmul(d1, d2.t())
    dotstemp = dots * torch.exp(temperature.to(dots.device))
    labels = torch.arange(len(d1)).to(d1.device)
    c = nn.CrossEntropyLoss()
    loss1 = c(dots, labels)
    loss2 = c(dots.t(), labels)
    loss1temp = c(dotstemp, labels)
    loss2temp = c(dotstemp.t(), labels)
    return (loss1 + loss2) / 2, (loss1temp + loss2temp) / 2


# the patient series discrimination objective
def patdis_objective(patdisembs: torch.Tensor,
                     map: torch.Tensor,
                     tau: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute patient discrimination objective loss."""
    device = patdisembs.device
    numseries = len(patdisembs)
    mask = torch.zeros(numseries, numseries).to(device)
    for h in set(map.tolist()):
        masked = (map == h).long().to(device)
        mask += torch.outer(masked, masked)
    mask.fill_diagonal_(0)
    masksum = mask.sum(dim=1)
    logits = torch.matmul(patdisembs, patdisembs.t()) / tau
    logits.fill_diagonal_(-10)
    q = torch.nn.functional.softmax(logits)
    aggscores = (mask * q).sum(dim=1)
    # the first term is loss, the second term is weighted aggregation score used to track overall patient discrimination performance
    return -torch.dot(aggscores.log(), 1.0 / masksum), torch.dot(
        aggscores, 1.0 / masksum) / (1.0 / masksum).sum()
