module Buffers

using Requires
using GPUifyLoops
using MPI

import ..haspkg

##
# Implements a double buffering for MPI communication
# MPI.Isend and MPI.Irecv operate on the secondary buffers
# and the primary buffers are used as a staging ground to
# scatter the data from. 
#
# Status:
# - Unoptimized host memory
# - Pinned host memory
#
# TODO:
# - Optimize this scheme for MPI-aware CUDA
#   should remove one of the buffers
# - Implement HostBuffer
# - Implement DeviceBuffer

export DoubleBuffer, BufferKind

@enum BufferKind begin
  Host
  Pinned
  HostBuffer
  DeviceBuffer
end

struct DoubleBuffer{T, Arr, Buff}
  primary   :: Arr
  secondary :: Buff

  function DoubleBuffer{T}(::Type{Arr}, kind, dims...) where {T, Arr}
    if kind == Host
      secondary = zeros(FT, dims...)
    elseif kind == Pinned
      secondary = zeros(FT, dims...)
      register(secondary)
      # XXX: Should this finalizer be attached to the DoubleBuffer struct?
      finalizer(unregister, secondary)
    else
      error("Bufferkind $bufferkind is not implemented yet")
    end

    buffer = new{T, typeof(primary), typeof(secondary)}(primary, secondary)

    return buffer
  end
end

register(x) = nothing
unregister(x) = nothing

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays
  using .CuArrays.CUDAnative
  import .CuArrays.CUDAdrv.Mem

  function register(arr) 
    GC.@preserve arr begin
      Mem.register(Mem.HostBuffer, pointer(arr), sizeof(arr), Mem.HOSTREGISTER_DEVICEMAP)
    end
  end

  function unregister(arr)
    GC.@preserve arr begin
      Mem.register(pointer(arr))
    end
  end
end

# MPI Buffer handling

function knl_fillsendbuf!(::Val{Np}, ::Val{nvar}, sendbuf, buf,
                          vmapsend, nvmapsend) where {Np, nvar}

  @inbounds @loop for i in (1:nvmapsend;
                            threadIdx().x + blockDim().x * (blockIdx().x-1))
    e, n = fldmod1(vmapsend[i], Np)
    @unroll for s = 1:nvar
      sendbuf[s, i] = buf[n, s, e]
    end
  end
  nothing
end

function _fillsendbuf!(sendbuf, buf, vmapsend)
  if length(vmapsend) > 0
    Np = size(buf, 1)
    nvar = size(buf, 2)

    threads = 256
    blocks = div(length(vmapsend) + threads - 1, threads)
    @launch(device(buf), threads=threads, blocks=blocks,
            knl_fillsendbuf!(Val(Np), Val(nvar), sendbuf, buf, vmapsend,
                             length(vmapsend)))
  end
end

function fillsendbuf!(host_sendbuf::Array, device_sendbuf::Array, buf::Array,
                      vmapsend::Array)
  _fillsendbuf!(host_sendbuf, buf, vmapsend)
end

function fillsendbuf!(host_sendbuf, device_sendbuf, buf, vmapsend)
  _fillsendbuf!(device_sendbuf, buf, vmapsend,)

  if length(vmapsend) > 0
    copyto!(host_sendbuf, device_sendbuf)
  end
end

function fillsendbuf!(buffer, data, vmapsend)
  fillsendbuf!(buffer.secondary, buffer.primary, data, vmapsend)
end

function knl_transferrecvbuf!(::Val{Np}, ::Val{nvar}, buf, recvbuf,
                              vmaprecv, nvmaprecv) where {Np, nvar}

  @inbounds @loop for i in (1:nvmaprecv;
                            threadIdx().x + blockDim().x * (blockIdx().x-1))
    e, n = fldmod1(vmaprecv[i], Np)
    @unroll for s = 1:nvar
      buf[n, s, e] = recvbuf[s, i]
    end
  end
  nothing
end

function _transferrecvbuf!(buf, recvbuf, vmaprecv)
  if length(vmaprecv) > 0
    Np = size(buf, 1)
    nvar = size(buf, 2)

    threads = 256
    blocks = div(length(vmaprecv) + threads - 1, threads)
    @launch(device(buf), threads=threads, blocks=blocks,
            knl_transferrecvbuf!(Val(Np), Val(nvar), buf, recvbuf,
                                 vmaprecv, length(vmaprecv)))
  end
end

function transferrecvbuf!(device_recvbuf::Array, host_recvbuf::Array,
                          buf::Array, vmaprecv::Array)
  _transferrecvbuf!(buf, host_recvbuf, vmaprecv)
end

function transferrecvbuf!(device_recvbuf, host_recvbuf, buf, vmaprecv)
  if length(vmaprecv) > 0
    copyto!(device_recvbuf, host_recvbuf)
  end

  _transferrecvbuf!(buf, device_recvbuf, vmaprecv)
end

function transferrecvbuf!(buffer::DoubleBuffer, data, vmaprecv)
  transferrecvbuf!(buffer.primary, buffer.secondary, data, vmaprecv)
end

end # module