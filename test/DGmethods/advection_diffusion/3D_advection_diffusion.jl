using MPI
using CLIMA
using Logging
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using LinearAlgebra
using Printf
using Dates
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.Mesh.Grids: EveryDirection, HorizontalDirection, VerticalDirection, min_node_distance

const ArrayType = CLIMA.array_type()

if !@isdefined integration_testing
  if length(ARGS) > 0
    const integration_testing = parse(Bool, ARGS[1])
  else
    const integration_testing =
      parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  end
end

const output = parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_OUTPUT","false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{α, β} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(::Pseudo1D{α, β}, aux::Vars,
                                  geom::LocalGeometry) where {α, β}
  # Direction of flow is n with magnitude α
  #aux.u = α * n
  aux.u = SVector(α, 0, 0)

  # diffusion of strength β in the n direction
  n = SVector(1 / 3, 1 / 3, 1 / 3)
  aux.D = β * n * n'
end

function initial_condition!(::Pseudo1D{α, β}, state, aux, x, t) where {α, β}
  #ξn = dot(n, x)
  @inbounds x1, x2, x3 = x[1], x[2], x[3]
  # ξT = SVector(x) - ξn * n
  #state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
  #state.ρ = sin(x1) ^ 4 * exp(-t)
  state.ρ = sin(x1 + x2 + x3 - α * t)  * exp(-β * t)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname)
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
  exactnames = statenames .* "_exact"

  writevtk(filename, Q, dg, statenames, Qe, exactnames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames..., exactnames...))

    @info "Done writing VTK: $pvtuprefix"
  end
end


function run(mpicomm, dim, topl, N, timeend, FT, direction,
             α, β, vtkdir, outputtime)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  dx = min_node_distance(grid)
  dt = dx ^ 2 / 10
  @info "time step" dt
  dt = outputtime / ceil(Int64, outputtime / dt)

  #timeend = 5dt
  timeend = FT(1)

  model = AdvectionDiffusion{dim}(Pseudo1D{α, β}())
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty(),
               direction=direction())

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end
  callbacks = (cbinfo,)
  if ~isnothing(vtkdir)
    # create vtk dir
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model, "advection_diffusion")

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime/dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(lsrk))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model,
                "advection_diffusion")
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=callbacks)

  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  errf
end

using Test
let
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 4
  base_num_elem = 4

  numlevels = 3
  α = 0
  β = 1

    for FT in (Float64,)
      result = zeros(FT, numlevels)
      for dim = 3
        for direction in (EveryDirection,)
          for l = 1:numlevels
            Ne = 2^(l-1) * base_num_elem
            xrange = range(FT(0); length=Ne+1, stop=FT(2pi))
            brickrange = (xrange, xrange, xrange)
            periodicity = ntuple(j->true, dim)
            topl = StackedBrickTopology(mpicomm, brickrange;
                                        periodicity = periodicity)

            timeend = 1
            outputtime = 1


            @info (ArrayType, FT, dim, direction)
            vtkdir = output ? "vtk_advection" *
                              "_poly$(polynomialorder)" *
                              "_dim$(dim)_$(ArrayType)_$(FT)_$(direction)" *
                              "_level$(l)" : nothing
            result[l] = run(mpicomm, dim, topl, polynomialorder,
                            timeend, FT, direction, α, β, vtkdir,
                            outputtime)
          end
          @info begin
            msg = ""
            for l = 1:numlevels-1
              rate = log2(result[l]) - log2(result[l+1])
              msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
            end
            msg
          end
        end
      end
    end
end

nothing

