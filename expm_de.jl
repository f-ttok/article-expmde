using LinearAlgebra
using KrylovKit
using Logging
using Printf
using SuiteSparse

"""
    om99_x(k, h; β=0.25)

Compute x(kh) where x(t) = π/h × t/(1 - exp(-2t -α(1-e⁻ᵗ) - β(1-eᵗ))).
"""
function om99_x(k, h; β=0.25)
    α = β / sqrt(1 + log(1+π/h)/(4h))
    t = k * h

    if abs(t)^3 > eps()
        u1 = α * expm1(-t)
        u2 = -β * expm1(t)
        u = -2t + u1 + u2
        return k * π / (-expm1(u))
    else
        c0 = 1 / (α + β + 2)
        c1 = 1 / 2 * (α^2 + 2 * α * β + 5α + β^2 + 3β + 4) / (α + β + 2)^2
        c2n = α^4 + 4 * α^3 * β + 8 * α^3 + 6 * α^2 * β^2 + 24 * α^2 * β + 25 * α^2 + 4 * α * β^3 + 24 * α * β^2 + 38 * α * β + 28 * α + β^4 + 8 * β^3 + 25 * β^2 + 28 * β + 16
        c2d = 12 * (α + β + 2)^3
        return π / h * (c0 + t * (c1 + t * c2n / c2d))
    end
end


"""
    om99_dx(k, h; β=0.25)

Compute x'(kh) where  x(t) = π/h × t/(1 - exp(-2t -α(1-e⁻ᵗ) - β(1-e⁻ᵗ))).
"""
function om99_dx(k, h; β=0.25)
    α = β / sqrt(1 + log(1 + π / h) / (4h))
    t = k * h
    if abs(t)^3 < eps()
        c0 = (α^2 + 2 * α * β + 5 * α + β^2 + 3 * β + 4) / (2 * (α^2 + 2 * α * β + 4 * α + β^2 + 4 * β + 4))
        c1n = α^4 + 4 * α^3 * β + 8 * α^3 + 6 * α^2 * β^2 + 24 * α^2 * β + 25 * α^2 + 4 * α * β^3 + 24 * α * β^2 + 38 * α * β + 28 * α + β^4 + 8 * β^3 + 25 * β^2 + 28 * β + 16
        c1d = 6 * α^3 + 18 * α^2 * β + 36 * α^2 + 18 * α * β^2 + 72 * α * β + 72 * α + 6 * β^3 + 36 * β^2 + 72 * β + 48
        c2n = -α^5 - 3 * α^4 * β - 8 * α^4 - 2 * α^3 * β^2 - 16 * α^3 * β - 24 * α^3 + 2 * α^2 * β^3 - 36 * α^2 * β - 36 * α^2 + 3 * α * β^4 + 16 * α * β^3 + 36 * α * β^2 - 12 * α + β^5 + 8 * β^4 + 24 * β^3 + 36 * β^2 + 12 * β
        c2d = 8 * α^4 + 32 * α^3 * β + 64 * α^3 + 48 * α^2 * β^2 + 192 * α^2 * β + 192 * α^2 + 32 * α * β^3 + 192 * α * β^2 + 384 * α * β + 256 * α + 8 * β^4 + 64 * β^3 + 192 * β^2 + 256 * β + 128
        return π / h * (c0 + t * (c1n / c1d + t * c2n / c2d))
    else
        u = -2t + α * expm1(-t) - β * expm1(t)
        du = -2 - α * exp(-t) - β * exp(t)
        if t < 0
            return π / h * (exp(-u) - 1 + t * du) / (2 * sinh(u / 2))^2
        else
            return π / h * (-expm1(u) + t * du * exp(u)) / expm1(u)^2
        end
    end
end


"""
Compute sin(x(kh)) where  x(t) = π/h × t/(1 - exp(-2t -α(1-e⁻ᵗ) - β(1-e⁻ᵗ))).
"""
function om99_sinx(k, h; β=0.25)
    α = β / sqrt(1 + log(1 + π / h) / (4h))
    t = k * h

    if abs(t) > 3
        u = -2t + α * expm1(-t) - β * expm1(t)
        if u > 0
            return sin(π * k / (-expm1(u)))
        else
            return (-1)^k * sin(k * π * exp(u) / (-expm1(u)))
        end
    else
        return sin(om99_x(k, h, β=β))
    end
end


"""
    exp_de(λ, h, ϵ=eps())

Computing the exponential of scalar by the DE formula
"""
function exp_de(λ, h, ϵ=eps())
    if real(λ) < 0
        l, r = get_interval(real(λ), ϵ, h)
    else
        l, r = Int(floor(-7 / h)), Int(floor(7 / h))
    end
    k = l:r
    x = om99_x.(k, h)
    dx = om99_dx.(k, h)
    sinx = om99_sinx.(k, h)
    return 2h * sum(@. dx * sinx * x / (x^2 + λ^2)) / π
end


mutable struct ExpmDEResult{T<:AbstractVecOrMat}
    X::T
    λ_right::Number
    l::Vector{Int64}
    r::Vector{Int64}
    h::Vector{Float64}
    errest::Vector{Float64}
end


"""
    expm_de(A[, h, B=I, V=I, ϵ=eps(), σ=-2.5, λ_right=nothing, lr=nothing, h0=0.2, hmin=1e-3])::ExpmDEResult

Computing the matrix exponential exp(B⁻¹A)V with the DE formula.
When h is given, the mesh size will be fixed to h.
When h is not given, the mesh size will be selected automatically.
The default values of both B and V are indentity matrices.

The optional parameters are as follows:

- `ϵ`: The tolerance for the truncation error, where the default value is eps() ≈ 1e-16.
- `σ`: The shift parameter, where the default value is -1/2.
- `λ_right`: The real value of the right most eigenvalue of B⁻¹A.
- `lr`: truncation points of the infinite sum (vector of integer).
- `h1`: The initial mesh size for the automatic quadrature.

If the two parameter `λ_right` is not set, it will be computed with KrylovKit package.

This function returns `ExpmDEResult` type.
"""
function expm_de(A, h; B=I, V=I, ϵ=eps(), σ=-2.5, λ_right=nothing, lr=nothing)::ExpmDEResult
    X = zero(A) * V
    result = ExpmDEResult(X, 0.0, zeros(Int,0), zeros(Int,0), zeros(0), zeros(0))

    λ_right = compute_λ_right!(λ_right, A, B)
    l, r = 0, 0
    if isnothing(lr)
        l, r = get_interval(abs(σ), ϵ, h)
    else
        l, r = minimum(lr), maximum(lr)
    end

    X .= compute_integral(A, B, V, h, l, r, σ, λ_right)
    result.λ_right = λ_right
    result.l, result.r, result.h = [l], [r], [h]
    result.X = X
    return result
end


function get_interval(σ, ϵ, h)
    r_max = Int(ceil(10 / h))
    r_min = 0
    r = 0
    while r_max - r_min >= 2
        r = Int(floor((r_max + r_min) / 2))
        est = err_right(h, r, σ)
        if est > ϵ / 2
            r_min = r
        else
            r_max = r
        end
    end
    r = r_max

    l_min = -Int(ceil(10 / h))
    l_max = 0
    l = 0
    while l_max - l_min >= 2
        l = Int(floor((l_min + l_max) / 2))
        est = err_left(h, l)
        if est > ϵ / 2
            l_max = l
        else
            l_min = l
        end
    end
    l = l_min

    return l, r
end


function err_right(h, r, σ; β=0.25)
    α = β / sqrt(1 + log(1 + π / h) / (4h))
    k = r+1:r+50
    kh = k * h
    v = @. -2kh + α * expm1(-kh) - β * expm1(kh)
    u = @. exp(v) / -expm1(v)
    est = 4 * π * (1+sqrt(2)) / abs(σ) * sum(@. k * u)
    return est
end


function err_left(h, l; β=0.25)
    k = l-50:l-1
    return h * sum(om99_dx.(k, h, β=β)) / π
end


function compute_λ_right!(λ_right, A, B)
    if isnothing(λ_right)
        n = size(A, 1)
        B_I = typeof(B) <: UniformScaling ? Diagonal(ones(n)) : B
        λ_right = geneigsolve((A, B_I), 1, :LR, tol=1e-4, verbosity=0)[1][1]
        if !isfinite(λ_right)
            error("failed for computing λ_right")
        end
    end
    return λ_right
end


function compute_integral(A, B, V, h, l, r, σ, λ_right)
    k = l:r
    x_list = om99_x.(k, h)
    dx_list = om99_dx.(k, h)
    sinx_list = om99_sinx.(k, h)

    X = zero(A) * V * im
    BV = B * V
    for (x, dx, sinx) in zip(x_list, dx_list, sinx_list)
        X += dx * sinx * im * (
                 (im * x * B + A + (σ - λ_right) * B) \ (BV)
                 -
                 (-im * x * B + A + (σ - λ_right) * B) \ (BV)
             )
    end
    return exp(-σ + λ_right) * (h / π) * X
end



function expm_de(A; h0=0.2, B=I, V=I, ϵ=eps(), σ=-2.5, λ_right=nothing, hmin=1e-3, η=10.0)
    X = zero(A) * V
    result = ExpmDEResult(X, 0.0, zeros(Int,0), zeros(Int,0), zeros(0), zeros(0))
    h = [h0, h0/2, h0/4]
    λ_right = compute_λ_right!(λ_right, A, B)
    result.λ_right = λ_right
    h1, h2, h3 = h
    l1, r1 = get_interval(abs(σ), ϵ/2, h1)
    X1 = compute_integral(A, B, V, h1, l1, r1, σ, λ_right)

    l2, r2 = get_interval(abs(σ), ϵ/2, h2)
    X2 = compute_integral(A, B, V, h2, l2, r2, σ, λ_right)

    l = [l1, l2]
    r = [r1, r2]
    errest = fill(Inf, 100)

    k_max = 5
    for k = 1:k_max
        l3, r3 = get_interval(abs(σ), ϵ/2, h3)
        push!(l, l3)
        push!(r, r3)
        X3 = compute_integral(A, B, V, h3, l3, r3, σ, λ_right)

        ϵ1 = norm(X1 - X3)
        ϵ2 = norm(X2 - X3)
        errest[k] = ϵ1
        errest[k+1] = ϵ2
        ρ = h1 * h2 * log(ϵ1 / ϵ2) / (h1 - h2)
        γ = ϵ1 * exp(ρ / h1)
        ϵ3 = γ * exp(-ρ / h3)
        errest[k+2] = ϵ3

        if ϵ3 < ϵ/η
            X .= X3
            break
        end

        h4 = ρ / log(γ / (ϵ/η))
        if ϵ2 >= 2*ϵ1 || h4 < hmin || !isfinite(h4)
            push!(h, h3/2)
            h1, h2, h3 = h2, h3, h3/2
            X1 .= X2
            X2 .= X3
            continue
        else
            push!(h, h4)
            l4, r4 = get_interval(abs(σ), ϵ/2, h4)
            push!(l, l4)
            push!(r, r4)
            push!(errest, ϵ)
            X .= compute_integral(A, B, V, h4, l4, r4, σ, λ_right)
            break
        end
    end

    result.h, result.r, result.l = h, r, l
    result.errest = filter(!isinf, errest)
    result.X = X
    return result
end