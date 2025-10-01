using Quromorphic
using Documenter

DocMeta.setdocmeta!(Quromorphic, :DocTestSetup, :(using Quromorphic); recursive=true)

makedocs(;
    modules=[Quromorphic],
    authors="jajapuramshivasai <jajapuramshivasai@gmail.com> and contributors",
    repo="https://github.com/Heterotic-Research/Quromorphic.jl/blob/{commit}{path}#{line}",
    sitename="Quromorphic.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Heterotic-Research.github.io/Quromorphic.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => [
            "Quantum Simulation" => "api/qsim.md",
            "Liquid State Machines" => "api/lsm.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/Heterotic-Research/Quromorphic.jl",
    devbranch="main",
)