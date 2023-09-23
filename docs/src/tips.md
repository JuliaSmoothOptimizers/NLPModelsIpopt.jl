## Linear solvers

To improve performance, Ipopt supports multiple linear solvers, the default one is [`MUMPS`](https://mumps-solver.org/)
but you can modify the linear solver with the keyword argument `linear_solver`.

### HSL

Obtain a license and download `HSL_jll.jl` from [https://licences.stfc.ac.uk/product/julia-hsl](https://licences.stfc.ac.uk/product/julia-hsl).

There are two versions available: LBT and OpenBLAS. LBT is the recommended option for Julia ≥ v1.9.

Install this download into your current environment using:
```julia
import Pkg
Pkg.develop(path = "/full/path/to/HSL_jll.jl")
```

We provide an example with the linear solvers `MA27` and `MA57`:
```julia
using HSL_jll, NLPModelsIpopt
stats_ma27 = ipopt(nlp, linear_solver="ma27")
stats_ma57 = ipopt(nlp, linear_solver="ma57")
```

### SPRAL

If you use NLPModelsIpopt.jl with Julia ≥ v1.9, the linear solver [SPRAL](https://github.com/ralna/spral) is available.
You can use it by setting the `linear_solver` attribute:
```julia
using NLPModelsIpopt
stats_spral = ipopt(nlp, linear_solver="spral")
```
Note that the following environment variables must be set before starting Julia:
```raw
export OMP_CANCELLATION=TRUE
export OMP_PROC_BIND=TRUE
```

## BLAS and LAPACK

With Julia v1.9 or later, Ipopt and the linear solvers [MUMPS](https://mumps-solver.org/index.php)
(default), [SPRAL](https://github.com/ralna/spral), and [HSL](https://licences.stfc.ac.uk/product/julia-hsl) are compiled with
[`libblastrampoline`](https://github.com/JuliaLinearAlgebra/libblastrampoline)
(LBT), a library that can change between BLAS and LAPACK backends at runtime.

The default BLAS and LAPACK backend is [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS).

Using LBT, we can also switch dynamically to other BLAS backends such as Intel
MKL and Apple Accelerate. Because Ipopt and the linear solvers heavily rely on
BLAS and LAPACK routines, using an optimized backend for a particular platform
can improve the performance.

### MKL

If you have [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) installed,
switch to MKL by adding `using MKL` to your code:

```julia
using MKL  # Replace OpenBLAS by Intel MKL
using NLPModelsIpopt
```

### AppleAccelerate

If you are using macOS ≥ v13.4 and you have [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl) installed, you can replace OpenBLAS as follows:

```julia
using AppleAccelerate  # Replace OpenBLAS by Apple Accelerate
using NLPModelsIpopt
```

### Display backends

Check what backends are loaded using:
```julia
import LinearAlgebra
LinearAlgebra.BLAS.lbt_get_config()
```
