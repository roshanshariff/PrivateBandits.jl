#!/usr/bin/env julia
#SBATCH --time=02:55:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --array=1-1920
#SBATCH --mail-user=rshariff@ualberta.ca
#SBATCH --mail-type=ALL

using JLD2
using OrderedCollections
using Printf

using PrivateBandits.DifferentialPrivacy
using PrivateBandits.LinearBandits
using PrivateBandits.Experiments

horizon = 5*10^7;
dp = (ε=1.0, δ=0.1)
env = EnvParams(dim=5, maxrewardmean=0.75, maxreward=1.0);

gaps = reverse!([0.1 .* 0.5.^(0:6); 0.0])

mechanisms = OrderedDict(
    "Gaussian" => GaussianMechanism,
    "Gaussian(Opt)" => OptShifted{GaussianMechanism}(env, horizon),
    "Wishart" => ShiftedWishart,
    "Wishart(Unshifted)" => WishartMechanism,
    "Wishart(Opt)" => OptShifted{WishartMechanism}(env, horizon)
)

(ρmin_lo, ρmin_hi) = (extrema ∘ map)(values(mechanisms)) do Mechanism
    strategy = make_strategy(env, horizon, Mechanism; dp...)
    regparams(strategy; α=1/2horizon).ρmin
end

num_ρmins = 8
ρmin_interp(c) = exp((1-c)*log(ρmin_lo) + c*log(ρmin_hi))
ρmins = ρmin_interp.(((1:num_ρmins).-2) .// (num_ρmins-3))

function task_params(A...; taskid=parse(Int, ENV["SLURM_ARRAY_TASK_ID"]))
    Tuple(CartesianIndices(((size.(A)...)..., taskid))[taskid])
end

(ρmin_ix, gap_ix, run_ix) = task_params(ρmins, gaps)
ρmin = ρmins[ρmin_ix]
gap = gaps[gap_ix]

algs = OrderedDict{Symbol, ContextLinBandit}(
    :NonPrivate => make_alg(env, horizon; ρ=ρmin),
    :Wishart => make_alg(env, horizon, shifted(WishartMechanism; ρmin=ρmin); dp...),
    :Gaussian => make_alg(env, horizon, shifted(GaussianMechanism; ρmin=ρmin); dp...)
)

arms = GapArms(env; gap=gap)

for (alg_name, alg) in algs
    result_name = "$alg_name,ρmin=$ρmin,Δ=$gap"
    mkpath(result_name)
    @time result = run_episode(env, alg, arms, horizon; subsample=10^4)
    @save joinpath(result_name, @sprintf("%03d.jld", run_ix)) result
end
