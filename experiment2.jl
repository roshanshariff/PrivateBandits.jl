#!/usr/bin/env julia
#SBATCH --time=02:30:00
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
env = EnvParams(dim=5, maxrewardmean=0.75, maxreward=1.0);
arms = GapArms(env; gap=0.5)

make_exp2_alg(mechanism) = make_private_alg(env, horizon, 1.0, 0.1, mechanism)

algs = OrderedDict{String, ContextLinBandit}(
    "Non-private" => make_alg(env, horizon; ρ=1.0),
    "Gaussian" => make_exp2_alg(GaussianMechanism),
    "Gaussian(Opt)" => make_exp2_alg(OptShifted{GaussianMechanism}(env, horizon)),
    "Wishart" => make_exp2_alg(ShiftedWishart),
    "Wishart(Unshifted)" => make_exp2_alg(WishartMechanism),
    "Wishart(Opt)" => make_exp2_alg(OptShifted{WishartMechanism}(env, horizon)),
)

taskid = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
(run, algid) = divrem(taskid-1, length(algs)) .+ 1
algname = collect(keys(algs))[algid]
alg = algs[algname]

mkpath(algname)
@time result = run_episode(env, alg, arms, horizon; subsample=10^4)
@save joinpath(algname, @sprintf("%03d.jld", run)) result
