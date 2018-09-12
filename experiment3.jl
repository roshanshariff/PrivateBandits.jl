#!/usr/bin/env julia
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --array=1-1080
#SBATCH --mail-user=rshariff@ualberta.ca
#SBATCH --mail-type=ALL

using JLD2
using DataStructures
using Printf
using Base.Iterators: product

using PrivateBandits.DifferentialPrivacy
using PrivateBandits.LinearBandits
using PrivateBandits.Experiments

horizon = 5*10^7;
dp = (ε=1.0, δ=0.1)
env = EnvParams(dim=5, maxrewardmean=0.75, maxreward=1.0);

gaps = range(0.0; stop=0.5, step=0.1)

(ρmin_lo, ρmin_hi) = map([ShiftedWishart, WishartMechanism]) do Mechanism
    strategy = make_strategy(env, horizon, Mechanism; dp...)
    regparams(strategy; α=1/2horizon).ρmin
end

ρmins = exp.(range(log(ρmin_lo); stop=log(ρmin_hi), length=6))

function task_params(A...; taskid=parse(Int, ENV["SLURM_ARRAY_TASK_ID"]))
    Tuple(CartesianIndices(((size.(A)...)..., taskid))[taskid])
end

(gap_ix, ρmin_ix, run_ix) = task_params(gaps, ρmins)
gap = gaps[gap_ix]
ρmin = ρmins[ρmin_ix]

arms = GapArms(env; gap=gap)
alg = make_alg(env, horizon, shifted(WishartMechanism; ρmin=ρmin); dp...)

algname = @sprintf("gap=%.1f,shift=%d:%d", gap, ρmin_ix-1, length(ρmins)-1)

mkpath(algname)
@time result = run_episode(env, alg, arms, horizon; subsample=10^4)
@save joinpath(algname, @sprintf("%03d.jld", run)) result
