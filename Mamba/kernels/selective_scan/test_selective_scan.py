# Modified by $@#Anonymous#@$ #20240123
# Copyright (C) 2023, Tri Dao, Albert Gu.

import math
import torch
import torch.nn.functional as F
import pytest
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from einops import rearrange, repeat
import time
from functools import partial

SSOFLEX_FLOAT = True


def build_selective_scan_fn(selective_scan_cuda: object = None, mode="mamba_ssm", tag=None):
    MODE = mode

    class SelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, nrows=1, backnrows=-1):
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if z is not None and z.stride(-1) != 1:
                z = z.contiguous()
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True
            if D is not None and (D.dtype != torch.float):
                ctx._d_dtype = D.dtype
                D = D.float()
            if delta_bias is not None and (delta_bias.dtype != torch.float):
                ctx._delta_bias_dtype = delta_bias.dtype
                delta_bias = delta_bias.float()

            assert u.shape[1] % (B.shape[1] * nrows) == 0 
            assert nrows in [1, 2, 3, 4] # 8+ is too slow to compile

            if backnrows > 0:
                assert u.shape[1] % (B.shape[1] * backnrows) == 0 
                assert backnrows in [1, 2, 3, 4] # 8+ is too slow to compile
            else:
                backnrows = nrows
            ctx.backnrows = backnrows
            
            if MODE in ["mamba_ssm"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
            elif MODE in ["ssoflex"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, SSOFLEX_FLOAT)
            elif MODE in ["sscore"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            elif MODE in ["sstest"]:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, nrows)
            elif MODE in ["sscorendstate"]:
                assert A.shape[-1] == 1 and B.shape[2] == 1 and C.shape[2] == 1
                A = A.view(-1)
                B = B.squeeze(2)
                C = C.squeeze(2)
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
            else:
                raise NotImplementedError

            ctx.delta_softplus = delta_softplus
            ctx.has_z = z is not None

            last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
            if not ctx.has_z:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
                return out if not return_last_state else (out, last_state)
            else:
                ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
                if MODE in ["mamba_ssm", "sstest"]:
                    out_z = rest[0]
                    return out_z if not return_last_state else (out_z, last_state)
                elif MODE in ["sscore", "ssoflex"]:
                    return out if not return_last_state else (out, last_state)

        @staticmethod
        def backward(ctx, dout, *args):
            if not ctx.has_z:
                u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
                z = None
                out = None
            else:
                u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
            # backward of selective_scan_cuda with the backward of chunk).
            # Here we just pass in None and dz will be allocated in the C++ code.
            if MODE in ["mamba_ssm"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
                    False # option to recompute out_z, not used here
                )
            elif MODE in ["sstest"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
                    False, ctx.backnrows  # option to recompute out_z, not used here
                )
            elif MODE in ["sscore", "ssoflex"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.backnrows
                )
            elif MODE in ["sscorendstate"]:
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                )
                dA = dA.unsqueeze(1)
                dB = dB.unsqueeze(2)
                dC = dC.unsqueeze(2)
            else:
                raise NotImplementedError
            
            dz = rest[0] if ctx.has_z else None
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            
            _dD = None
            if D is not None:
                if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                    _dD = dD.to(ctx._d_dtype)
                else:
                    _dD = dD

            _ddelta_bias = None
            if delta_bias is not None:
                if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                    _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
                else:
                    _ddelta_bias = ddelta_bias

            return (du, ddelta, dA, dB, dC,
                        dD if D is not None else None,
                        dz,
                        ddelta_bias if delta_bias is not None else None,
                        None, None, None, None)

    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, nrows=1, backnrows=-1):
        """if return_last_state is True, returns (out, last_state)
        last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
        not considered in the backward pass.
        """
        outs = SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, nrows, backnrows)
        if mode in ["ssoflex"]:
            return outs.to(u.dtype) if not return_last_state else (outs[0].to(u.dtype), outs[1])
        else:
            return outs

    selective_scan_fn.__repr__ = lambda *_ :f"selective_scan_fn | {mode} | {tag}"

    return selective_scan_fn


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_ref_v2(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    A = A.to(dtype_in)
    B = B.to(dtype_in)
    C = C.to(dtype_in)
    D = D.to(dtype_in) if D is not None else None
    z = z.to(dtype_in) if z is not None else None
    delta = delta.to(dtype_in) if delta is not None else None
    delta_bias = delta_bias.to(dtype_in) if delta_bias is not None else None

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B, "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C, "... (L two) -> ... L two", two=2))
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state.float())


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, *args, **kwargs):
    return selective_scan_ref_v2(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

# MODE = None
# MODE = "mamba_ssm"
# MODE = "sscore"
# MODE = "ssoflex"
# MODE = "sstest"
# MODE = "mamba_ssm_sscore" # 1344 items pass
# MODE = "mamba_ssm_sscorendstate" # 1344 items pass
MODE = "mamba_ssm_ssoflex" # 1344 items pass

if MODE in ["mamba_ssm"]:
    import selective_scan_cuda
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda, mode=MODE)
    selective_scan_ref = selective_scan_ref
elif MODE in ["ssoflex"]:
    import selective_scan_cuda_oflex
    selective_scan_cuda = selective_scan_cuda_oflex
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_oflex, mode=MODE)
    selective_scan_ref = selective_scan_ref
elif MODE in ["sscore"]:
    import selective_scan_cuda_core
    selective_scan_cuda = selective_scan_cuda_core
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_core, mode=MODE)
    selective_scan_ref = selective_scan_ref
elif MODE in ["sstest"]:
    import selective_scan_cuda_test
    selective_scan_cuda = selective_scan_cuda_test
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_test, mode=MODE)
    selective_scan_ref = selective_scan_ref
elif MODE in ["mamba_ssm_sscore"]:
    import selective_scan_cuda_core
    import selective_scan_cuda
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_core, mode="sscore")
    selective_scan_ref = build_selective_scan_fn(selective_scan_cuda, mode="mamba_ssm")
elif MODE in ["mamba_ssm_sstest"]:
    import selective_scan_cuda_test
    import selective_scan_cuda
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_test, mode="sstest")
    selective_scan_ref = build_selective_scan_fn(selective_scan_cuda, mode="mamba_ssm")
elif MODE in ["mamba_ssm_sscorendstate"]:
    import selective_scan_cuda_core
    import selective_scan_cuda
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_core, mode="sscorendstate")
    selective_scan_ref = build_selective_scan_fn(selective_scan_cuda, mode="mamba_ssm")
elif MODE in ["mamba_ssm_ssoflex"]:
    import selective_scan_cuda_oflex
    import selective_scan_cuda
    selective_scan_fn = build_selective_scan_fn(selective_scan_cuda_oflex, mode="ssoflex")
    selective_scan_ref = build_selective_scan_fn(selective_scan_cuda, mode="mamba_ssm")
else:
    selective_scan_cuda = None


print("use MODE:", MODE)
DSTATE = [1]
DIM = [768]
BATCHSIZE = [2]
# DSTATE = [1] if MODE in ["mamba_ssm_sscorendstate", "sscorendstate"] else [8]
NROWS = [1,2,3,4]
IDTYPE = MODE in [None]

# @pytest.mark.parametrize('wtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('wtype', [torch.float32])
@pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize('has_delta_bias', [False, True])
@pytest.mark.parametrize('delta_softplus', [False, True])
# @pytest.mark.parametrize('has_z', [False, True])
@pytest.mark.parametrize('has_z', [False])
@pytest.mark.parametrize('has_D', [False, True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
# @pytest.mark.parametrize("is_variable_C", [False, True])
@pytest.mark.parametrize("is_variable_C", [True])
# @pytest.mark.parametrize("is_variable_B", [False, True])
@pytest.mark.parametrize("is_variable_B", [True])
@pytest.mark.parametrize("nrows", NROWS)
@pytest.mark.parametrize("batch_size", BATCHSIZE)
@pytest.mark.parametrize("dim", DIM)
@pytest.mark.parametrize("dstate", DSTATE)
def test_selective_scan(is_variable_B, is_variable_C, varBC_groups, has_D, has_z, has_delta_bias,
                        delta_softplus, return_last_state, seqlen, itype, wtype, nrows, batch_size, dim, dstate):
    wtype = itype if IDTYPE else wtype
    print(f'method: {selective_scan_cuda}')
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 2
    # dim = 24
    # dstate = 8
    is_complex = wtype == torch.complex64
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype,
                    requires_grad=True)
    if not is_variable_C:
        C_shape = (dim, dstate)
    elif varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        C_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype,
                    requires_grad=True)
    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
    if has_z:
        z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    else:
        z = None
    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
    else:
        delta_bias = None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    out, *rest = selective_scan_fn(
        u, delta, A, B, C, D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
        return_last_state=return_last_state, nrows=nrows
    )
    if return_last_state:
        state = rest[0]
    out_ref, *rest = selective_scan_ref(
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )
    if return_last_state:
        state_ref = rest[0]
    # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    if return_last_state:
        print(f'State max diff: {(state - state_ref).abs().max().item()}')
        assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
    print(f'ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    if has_D:
        print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    if has_z:
        print(f'dz max diff: {(z.grad - z_ref.grad).abs().max().item()}')
    if has_delta_bias:
        print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')

    assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
                          atol=atolw if not is_variable_B else atol)
    assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
                          atol=atolw if not is_variable_C else atol)
    if has_D:
        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_z:
        assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
    if has_delta_bias:
        assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)

# test_selective_scan(True, True, 2, True, False, True, True, True, 64, torch.float32, torch.float32, 1, 2, 24, 1)

