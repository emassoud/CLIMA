using .NumericalFluxes: CentralHyperGradPenalty, CentralHyperGradFlux, CentralHyperDivPenalty, CentralHyperDivFlux
using LinearAlgebra
using ..Mesh.Grids

struct DGModel{BL,G,NFND,NFD,GNF,HGNF,AS,DS,HDS,D,MD}
  balancelaw::BL
  grid::G
  numfluxnondiff::NFND
  numfluxdiff::NFD
  gradnumflux::GNF
  hypergradnumflux::HGNF
  auxstate::AS
  diffstate::DS
  hyperdiffstate::HDS
  direction::D
  modeldata::MD
end
function DGModel(balancelaw, grid, numfluxnondiff, numfluxdiff, gradnumflux;
                 auxstate=create_auxstate(balancelaw, grid),
                 diffstate=create_diffstate(balancelaw, grid),
                 hyperdiffstate=create_hyperdiffstate(balancelaw, grid),
                 direction=EveryDirection(), modeldata=nothing)
  # FIXME
  hypergradnumflux = CentralHyperGradPenalty()
  DGModel(balancelaw, grid,
          numfluxnondiff, numfluxdiff, gradnumflux,
          hypergradnumflux,
          auxstate, diffstate, hyperdiffstate, direction, modeldata)
end

function (dg::DGModel)(dQdt, Q, ::Nothing, t; increment=false)
  bl = dg.balancelaw
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Nfp = Nq * Nqk
  nrealelem = length(topology.realelems)

  Qvisc = dg.diffstate
  Qhypervisc_grad, Qhypervisc_div = dg.hyperdiffstate
  auxstate = dg.auxstate

  FT = eltype(Q)
  nviscstate = num_diffusive(bl, FT)
  nhyperviscstate = num_hyperdiffusive(bl, FT)

  lgl_weights_vec = grid.ω
  Dmat = grid.D
  vgeo = grid.vgeo
  sgeo = grid.sgeo
  vmapM = grid.vmapM
  vmapP = grid.vmapP
  elemtobndy = grid.elemtobndy
  polyorder = polynomialorder(dg.grid)

  Np = dofs_per_element(grid)
  
  #x1 = vgeo[:, Grids._x1, :]
  #x2 = vgeo[:, Grids._x2, :]
  #x3 = vgeo[:, Grids._x3, :]

  communicate = !(isstacked(topology) &&
                  typeof(dg.direction) <: VerticalDirection)

  update_aux!(dg, bl, Q, t)

  ########################
  # Gradient Computation #
  ########################
  if communicate
    MPIStateArrays.start_ghost_exchange!(Q)
    MPIStateArrays.start_ghost_exchange!(auxstate)
  end

  if nviscstate > 0 || nhyperviscstate > 0

    @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
            volumeviscterms!(bl, Val(dim), Val(polyorder), dg.direction, Q.data,
                             Qvisc.data, Qhypervisc_grad.data, auxstate.data, vgeo, t, Dmat,
                             topology.realelems))

    if communicate
      MPIStateArrays.finish_ghost_recv!(Q)
      MPIStateArrays.finish_ghost_recv!(auxstate)
    end

    @launch(device, threads=Nfp, blocks=nrealelem,
            faceviscterms!(bl, Val(dim), Val(polyorder), dg.direction,
                           dg.gradnumflux, dg.hypergradnumflux,
                           Q.data, Qvisc.data, Qhypervisc_grad.data, auxstate.data,
                           vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                           topology.realelems))

    #println("First grad")
    #@show maximum(abs.(Qhypervisc_grad[:, 1, :]))

    communicate && MPIStateArrays.start_ghost_exchange!(Qvisc)
  end
  

  #@show Qhypervisc_div[:]
  if nhyperviscstate > 0

    #########################
    # Laplacian Computation #
    #########################
   
    # STRONG
    @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
            volumehyperviscterms_a_strong!(bl, Val(dim), Val(polyorder), dg.direction,
                                           Qhypervisc_grad.data, Qhypervisc_div.data, vgeo, Dmat,
                                           topology.realelems))
    
    @launch(device, threads=Nfp, blocks=nrealelem,
            facehyperviscterms_a_strong!(bl, Val(dim), Val(polyorder), dg.direction,
                                         #dg.hypergradnumflux,
                                         CentralHyperDivPenalty(),
                                         Qhypervisc_grad.data, Qhypervisc_div.data,
                                         vgeo, sgeo, vmapM, vmapP, elemtobndy,
                                         topology.realelems))
    
    # WEAK
    #@launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
    #        volumehyperviscterms_a_weak!(bl, Val(dim), Val(polyorder), dg.direction,
    #                                     Qhypervisc_grad.data, Qhypervisc_div.data, vgeo, Dmat,
    #                                     topology.realelems))
    #
    #@launch(device, threads=Nfp, blocks=nrealelem,
    #        facehyperviscterms_a_weak!(bl, Val(dim), Val(polyorder), dg.direction,
    #                                   #dg.hypergradnumflux,
    #                                   CentralHyperDivFlux(),
    #                                   Qhypervisc_grad.data, Qhypervisc_div.data,
    #                                   vgeo, sgeo, vmapM, vmapP, elemtobndy,
    #                                   topology.realelems))
    
    #println("First div")
    #@show maximum(abs.(Qhypervisc_div[:, 1, :]))
    #@show maximum(abs.(Qhypervisc_div[:, 1, :] + 3 .* sin.(x1 + x2 + x3) .* exp(-t / 100)))

    #####################################
    # Gradient of Laplacian Computation #
    #####################################
   
    #println("before hypervisc volume")
    #@show maximum(abs.(Qhypervisc_grad[:, 2, :]))
    #@show maximum(abs.(Qhypervisc_grad[:, 3, :]))
   
    # WEAK
    @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
            volumehyperviscterms_b_weak!(bl, Val(dim), Val(polyorder), dg.direction,
                                    Qhypervisc_grad.data, Qhypervisc_div.data,
                                    Q.data, auxstate.data,
                                    vgeo, Dmat,
                                    topology.realelems, t))
    

    @launch(device, threads=Nfp, blocks=nrealelem,
            facehyperviscterms_b_weak!(bl, Val(dim), Val(polyorder), dg.direction,
                                      #dg.hypergradnumflux,
                                      #CentralHyperGradPenalty(),
                                      CentralHyperGradFlux(),
                                      Qhypervisc_grad.data, Qhypervisc_div.data,
                                      Q.data, auxstate.data,
                                      vgeo, sgeo, vmapM, vmapP, elemtobndy,
                                      topology.realelems, t))
   
    # STRONG
    #@launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
    #        volumehyperviscterms_b_strong!(bl, Val(dim), Val(polyorder), dg.direction,
    #                                Qhypervisc_grad.data, Qhypervisc_div.data, vgeo, Dmat,
    #                                topology.realelems))
    #

    #@launch(device, threads=Nfp, blocks=nrealelem,
    #        facehyperviscterms_b_strong!(bl, Val(dim), Val(polyorder), dg.direction,
    #                                  #dg.hypergradnumflux,
    #                                  #CentralHyperGradPenalty(),
    #                                  CentralHyperGradPenalty(),
    #                                  Qhypervisc_grad.data, Qhypervisc_div.data,
    #                                  vgeo, sgeo, vmapM, vmapP, elemtobndy,
    #                                  topology.realelems))
    #println("Second grad")
    #@show maximum(abs.(Qhypervisc_grad[:, 1, :]))

    #@show maximum(abs.(Qhypervisc_grad[:, 1, :] .+ 3 .* cos.(x1 + x2 + x3) .* exp(-t / 100)))
    #@show maximum(abs.(Qhypervisc_grad[:, 2, :] .+ 3 .* cos.(x1 + x2 + x3) .* exp(-t / 100)))
    #@show maximum(abs.(Qhypervisc_grad[:, 3, :] .+ 3 .* cos.(x1 + x2 + x3) .* exp(-t / 100)))
    #@show maximum(abs.(Qhypervisc_grad[:, 1, :]))
    #@show maximum(abs.(Qhypervisc_grad[:, 2, :] .+ 3 // 100 .* cos.(x1 + x2 + x3) .* exp(-3t / 100)))
    #@show maximum(abs.(Qhypervisc_grad[:, 3, :]))
  end

  #Qhypervisc_div .*= -1 / 3
  #if t > 0.99
  # @show euclidean_distance(Qhypervisc_div, Q)
  #end
  #@show Qhypervisc_div[:]
  

  ###################
  # RHS Computation #
  ###################
  @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
          volumerhs!(bl, Val(dim), Val(polyorder), dg.direction, dQdt.data,
                     Q.data, Qvisc.data, Qhypervisc_grad.data, auxstate.data, vgeo, t,
                     lgl_weights_vec, Dmat, topology.realelems, increment))

  if communicate
    if nviscstate > 0
      MPIStateArrays.finish_ghost_recv!(Qvisc)
    else
      MPIStateArrays.finish_ghost_recv!(Q)
      MPIStateArrays.finish_ghost_recv!(auxstate)
    end
  end

  @launch(device, threads=Nfp, blocks=nrealelem,
          facerhs!(bl, Val(dim), Val(polyorder), dg.direction,
                   dg.numfluxnondiff,
                   dg.numfluxdiff,
                   dQdt.data, Q.data, Qvisc.data, Qhypervisc_grad.data,
                   auxstate.data, vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
                   topology.realelems))

  # Just to be safe, we wait on the sends we started.
  if communicate
    MPIStateArrays.finish_ghost_send!(Qvisc)
    MPIStateArrays.finish_ghost_send!(Q)
  end
end

function init_ode_state(dg::DGModel, args...;
                        forcecpu=false,
                        commtag=888)
  device = arraytype(dg.grid) <: Array ? CPU() : CUDA()

  bl = dg.balancelaw
  grid = dg.grid

  state = create_state(bl, grid, commtag)

  topology = grid.topology
  Np = dofs_per_element(grid)

  auxstate = dg.auxstate
  dim = dimensionality(grid)
  polyorder = polynomialorder(grid)
  vgeo = grid.vgeo
  nrealelem = length(topology.realelems)

  if !forcecpu
    @launch(device, threads=(Np,), blocks=nrealelem,
            initstate!(bl, Val(dim), Val(polyorder), state.data, auxstate.data, vgeo,
                     topology.realelems, args...))
  else
    h_vgeo = Array(vgeo)
    h_state = similar(state, Array)
    h_auxstate = similar(auxstate, Array)
    h_auxstate .= auxstate
    @launch(CPU(), threads=(Np,), blocks=nrealelem,
      initstate!(bl, Val(dim), Val(polyorder), h_state.data, h_auxstate.data, h_vgeo,
          topology.realelems, args...))
    state .= h_state
  end

  MPIStateArrays.start_ghost_exchange!(state)
  MPIStateArrays.finish_ghost_exchange!(state)

  return state
end

function indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                    Q::MPIStateArray, auxstate::MPIStateArray,
                                    t::Real)

  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  FT = eltype(Q)

  vgeo = grid.vgeo
  polyorder = polynomialorder(dg.grid)

  # do integrals
  nintegrals = num_integrals(m, FT)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_indefinite_stack_integral!(m, Val(dim), Val(polyorder),
                                         Val(nvertelem), Q.data, auxstate.data,
                                         vgeo, grid.Imat, 1:nhorzelem,
                                         Val(nintegrals)))
end

# fallback
function update_aux!(dg::DGModel, bl::BalanceLaw, Q::MPIStateArray, t::Real)
end

function reverse_indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                            auxstate::MPIStateArray, t::Real)

  device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  FT = eltype(auxstate)

  vgeo = grid.vgeo
  polyorder = polynomialorder(dg.grid)

  # do integrals
  nintegrals = num_integrals(m, FT)
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_reverse_indefinite_stack_integral!(Val(dim), Val(polyorder),
                                                 Val(nvertelem), auxstate.data,
                                                 1:nhorzelem,
                                                 Val(nintegrals)))
end

function nodal_update_aux!(f!, dg::DGModel, m::BalanceLaw, Q::MPIStateArray,
                           t::Real)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  nrealelem = length(topology.realelems)

  polyorder = polynomialorder(dg.grid)

  Np = dofs_per_element(grid)

  ### update aux variables
  @launch(device, threads=(Np,), blocks=nrealelem,
          knl_nodal_update_aux!(m, Val(dim), Val(polyorder), f!,
                          Q.data, dg.auxstate.data, dg.diffstate.data, t,
                          topology.realelems))
end

function copy_stack_field_down!(dg::DGModel, m::BalanceLaw,
                                auxstate::MPIStateArray, fldin, fldout)

  device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  DFloat = eltype(auxstate)

  vgeo = grid.vgeo
  polyorder = polynomialorder(dg.grid)

  # do integrals
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_copy_stack_field_down!(Val(dim), Val(polyorder), Val(nvertelem),
                                     auxstate.data, 1:nhorzelem, Val(fldin),
                                     Val(fldout)))
end

function MPIStateArrays.MPIStateArray(dg::DGModel, commtag=888)
  bl = dg.balancelaw
  grid = dg.grid

  state = create_state(bl, grid, commtag)

  return state
end
