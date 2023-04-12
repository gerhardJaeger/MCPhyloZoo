
using MCPhylo
using Revise
using Random;
Random.seed!(1234);

##
tree, data = make_tree_with_data("Example.nex");

data_dictionary = Dict{Symbol, Any}(
  :data => data
);

##

model = Model(
    data=Stochastic(
        3,
        (tree, eq_freq, rates) -> PhyloDist(tree, eq_freq, [1.0], rates, Restriction),
        false
    ),
    eq_freq=Stochastic(
        1,
        () -> Dirichlet(2, 1)
    ),
    tree=Stochastic(
        Node(),
        () -> TreeDistribution(CompoundDirichlet(1.0, 1.0, 0.100, 1.0)),
        true
    ),
    rates=Logical(1, (a) -> discrete_gamma_rates(a, a, 4)),
    a=Stochastic(() -> Exponential(), true)
)
##

scheme = [PNUTS(:tree, target=0.7, targetNNI=0.5),
    #Slice(:tree, 0.1),Slice(:tree, 1.0),Slice(:tree, 10.),
    RWM(:tree, [:NNI, :SPR]),
    SliceSimplex(:eq_freq),
    Slice(:a, 1.0)
]

setsamplers!(model, scheme);

inits = [
    Dict{Symbol,Union{Any,Real}}(
        :tree => tree,
        :eq_freq => rand(Dirichlet(2, 1)),
        :data => data_dictionary[:data],
        :a => rand()
    ) for c in 1:2
];

##

sim = mcmc(
    model,
    data_dictionary,
    inits,
    5000,
    burnin=2500,
    thin=5,
    chains=2,
    trees=true,
    verbose=false
)

gelmandiag(sim)

plot(sim)

describe(sim)