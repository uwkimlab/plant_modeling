using Cropbox

include("root.jl")

root_maize = (
    :RootArchitecture => :maxB => 5,
    :BaseRoot => :T => [
        # P F S
          0 1 0 ; # P
          0 0 1 ; # F
          0 0 0 ; # S
    ],
    :PrimaryRoot => (
        :lb => 0.1 ± 0.01,
        :la => 18.0 ± 1.8,
        :ln => 0.6 ± 0.06,
        :lmax => 89.7 ± 7.4,
        :r => 6.0 ± 0.6,
        :Δx => 0.5,
        :σ => 10,
        :θ => 80 ± 8,
        :N => 1.5,
        :a => 0.04 ± 0.004,
        :color => Root.RGBA(1, 0, 0, 1),
    ),
    :FirstOrderLateralRoot => (
        :lb => 0.2 ± 0.04,
        :la => 0.4 ± 0.04,
        :ln => 0.4 ± 0.03,
        :lmax => 0.6 ± 1.6,
        :r => 2.0 ± 0.2,
        :Δx => 0.1,
        :σ => 20,
        :θ => 70 ± 15,
        :N => 1,
        :a => 0.03 ± 0.003,
        :color => Root.RGBA(0, 1, 0, 1),
    ),
    :SecondOrderLateralRoot => (
        :lb => 0,
        :la => 0.4 ± 0.02,
        :ln => 0,
        :lmax => 0.4,
        :r => 2.0 ± 0.2,
        :Δx => 0.1,
        :σ => 20,
        :θ => 70 ± 10,
        :N => 2,
        :a => 0.02 ± 0.002,
        :color => Root.RGBA(0, 0, 1, 1),
    )
)

container_pot = :Pot => (
    :r1 => 10,
    :r2 => 6,
    :height => 30,
)
soilcore = :SoilCore => (
    :d => 5,
    :l => 20,
    :x0 => 3,
    :y0 => 3,
)

b = instance(Root.Pot, config=container_pot)
s = instance(Root.RootArchitecture; config=root_maize, options=(; box=b), seed=0)
r = simulate!(s, stop=500)
# Root.render(s)
