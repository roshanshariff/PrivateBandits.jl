#!/usr/bin/env julia

using JLD2

using PrivateBandits.Experiments


function loadresults(dirname)
    results = Dict{String, Vector{ExpResult}}()
    for dir in filter(isdir, readdir(dirname))
        resultslist = ExpResult[]
        for filename in readdir(dir)
            filepath = joinpath(dir, filename)
            endswith(filename, ".jld") &&
                isfile(filepath) &&
                filesize(filepath) > 0 ||
                continue
            try
                @show filepath
                @load filepath result
                push!(resultslist, result)
            catch err
                showerror(STDERR, err)
            end
        end
        results[dir] = resultslist
    end
    return results
end

results = loadresults(".")
@save "results.jld" results

for (key, val) in results
    println("$key: $(length(val)) results")
end
