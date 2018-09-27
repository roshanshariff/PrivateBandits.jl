#!/usr/bin/env julia
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --array=1-200
#SBATCH --mail-user=rshariff@ualberta.ca
#SBATCH --mail-type=ALL

using JLD2
using Printf

using PrivateBandits.LinearBandits
using PrivateBandits.Experiments

horizon = 10^5
gap = 0.1
dims = round.(Int, exp2.(range(log2(4); stop=log2(64), length=17)))

envs = let base_env = EnvParams(dim=0, maxrewardmean=0.75);
    [:GaussianReward => EnvParams(base_env; σ=1.0),
     :BernoulliReward => EnvParams(base_env; maxreward=1.0)]
end

function task_params(A...; taskid=parse(Int, ENV["SLURM_ARRAY_TASK_ID"]))
    Tuple(CartesianIndices(((size.(A)...)..., taskid))[taskid])
end

(env_ix, run_ix) = task_params(envs)

@time for dim in dims
    (env_name, base_env) = envs[env_ix]
    env = EnvParams(base_env; dim=dim)
    alg = make_alg(env, horizon; ρ=1.0)
    arms = GapArms(env; gap=gap)
    result_name = @sprintf("%s,dim=%02d", env_name, dim)
    mkpath(result_name)
    result = run_episode(env, alg, arms, horizon; subsample=10^4)
    @save joinpath(result_name, @sprintf("%03d.jld", run_ix)) result
end
