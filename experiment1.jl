#!/usr/bin/env julia
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=1000M
#SBATCH --array=1-510
#SBATCH --mail-user=rshariff@ualberta.ca
#SBATCH --mail-type=ALL

using JLD2
using DataStructures
using Printf

using PrivateBandits.DifferentialPrivacy
using PrivateBandits.LinearBandits
using PrivateBandits.Experiments

(run, dim) = begin
    dims = round.(Int, 2 .^ LinRange(log2(4), log2(64), 17))
    taskid = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    (run, dim) = divrem(taskid-1, length(dims))
    (run+1, dims[dim+1])
end

horizon = 10^5;
env = EnvParams(dim=dim, maxrewardmean=0.75, σ=1.0);
arms = GapArms(env; gap=0.5)

alg = make_alg(env, horizon; ρ=1.0)

resultdir = @sprintf("dim=%02d", dim)
mkpath(resultdir)
@time result = run_episode(env, alg, arms, horizon; subsample=10^4)
@save joinpath(resultdir, @sprintf("%03d.jld", run)) result
