#!/usr/bin/env julia
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --array=1-210
#SBATCH --mail-user=rshariff@ualberta.ca
#SBATCH --mail-type=ALL

using JLD2
using DataStructures
using Printf

using PrivateBandits.DifferentialPrivacy
using PrivateBandits.LinearBandits
using PrivateBandits.Experiments

horizon = 5*10^7;
dp = (ε=1.0, δ=0.1)
env = EnvParams(dim=5, maxrewardmean=0.75, maxreward=1.0);

gaps = [0.1]

make_private_alg(Mechanism) = make_alg(env, horizon, Mechanism; dp...)

algs = OrderedDict{Symbol, ContextLinBandit}(
    :NonPrivate => make_alg(env, horizon; ρ=1.0),
    :Gaussian => make_private_alg(GaussianMechanism),
    :GaussianOpt => make_private_alg(OptShifted{GaussianMechanism}(env, horizon)),
    :Wishart => make_private_alg(ShiftedWishart),
    :WishartUnshifted => make_private_alg(WishartMechanism),
    :WishartOpt => make_private_alg(OptShifted{WishartMechanism}(env, horizon)),
)

alg_names = collect(keys(algs))

function task_params(A...; taskid=parse(Int, ENV["SLURM_ARRAY_TASK_ID"]))
    Tuple(CartesianIndices(((size.(A)...)..., taskid))[taskid])
end

(alg_ix, gap_ix, run_ix) = task_params(alg_names, gaps)
alg_name = alg_names[alg_ix]
alg = algs[alg_name]
gap = gaps[gap_ix]
arms = GapArms(env; gap=gap)

result_name = "$alg_name,Δ=$gap"
mkpath(result_name)
@time result = run_episode(env, alg, arms, horizon; subsample=10^4)
@save joinpath(result_name, @sprintf("%03d.jld", run_ix)) result
