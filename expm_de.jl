using LinearAlgebra
import LinearAlgebra.eigvals

"""
    om99_x(k, h; β=0.25)

Compute x(kh) where x(t) = π/h × t/(1 - exp(-2t -α(1-e⁻ᵗ) - β(1-eᵗ))).
"""
function om99_x(k, h; β=0.25)
    α = β / sqrt(1 + log(1 + π / h) / (4h))
    t = k * h

    if abs(t)^3 > eps()
        u1 = α * expm1(-t)
        u2 = -β * expm1(t)
        u = -2t + u1 + u2
        return k * π / (-expm1(u))
    else
        c0 = 1 / (α + β + 2)
        c1 = 1 / 2 * (α^2 + 2 * α * β + 5α + β^2 + 3β + 4) / (α + β + 2)^2
        c2n =
            α^4 +
            4 * α^3 * β +
            8 * α^3 +
            6 * α^2 * β^2 +
            24 * α^2 * β +
            25 * α^2 +
            4 * α * β^3 +
            24 * α * β^2 +
            38 * α * β +
            28 * α +
            β^4 +
            8 * β^3 +
            25 * β^2 +
            28 * β +
            16
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
        c0 =
            (α^2 + 2 * α * β + 5 * α + β^2 + 3 * β + 4) /
            (2 * (α^2 + 2 * α * β + 4 * α + β^2 + 4 * β + 4))
        c1n =
            α^4 +
            4 * α^3 * β +
            8 * α^3 +
            6 * α^2 * β^2 +
            24 * α^2 * β +
            25 * α^2 +
            4 * α * β^3 +
            24 * α * β^2 +
            38 * α * β +
            28 * α +
            β^4 +
            8 * β^3 +
            25 * β^2 +
            28 * β +
            16
        c1d =
            6 * α^3 +
            18 * α^2 * β +
            36 * α^2 +
            18 * α * β^2 +
            72 * α * β +
            72 * α +
            6 * β^3 +
            36 * β^2 +
            72 * β +
            48
        c2n =
            -α^5 - 3 * α^4 * β - 8 * α^4 - 2 * α^3 * β^2 - 16 * α^3 * β - 24 * α^3 +
            2 * α^2 * β^3 - 36 * α^2 * β - 36 * α^2 +
            3 * α * β^4 +
            16 * α * β^3 +
            36 * α * β^2 - 12 * α +
            β^5 +
            8 * β^4 +
            24 * β^3 +
            36 * β^2 +
            12 * β
        c2d =
            8 * α^4 +
            32 * α^3 * β +
            64 * α^3 +
            48 * α^2 * β^2 +
            192 * α^2 * β +
            192 * α^2 +
            32 * α * β^3 +
            192 * α * β^2 +
            384 * α * β +
            256 * α +
            8 * β^4 +
            64 * β^3 +
            192 * β^2 +
            256 * β +
            128
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
        return sin(om99_x(k, h; β=β))
    end
end

mutable struct ExpmDEResult{T<:AbstractVecOrMat}
    X::T
    σ::Number
    l::Vector{Int64}
    r::Vector{Int64}
    h::Vector{Float64}
    errest::Vector{Float64}
end

function err_left(h, l; β=0.25)
    k = (l - 50):(l - 1)
    return 2h * sum(om99_dx.(k, h, β=β)) / π
end

function err_right(h, r, nrm; β=0.25)
    α = β / sqrt(1 + log(1 + π / h) / (4h))
    k = (r + 1):(r + 50)
    kh = k * h
    u = @. exp(-2kh + α * expm1(-kh) - β * expm1(kh))
    est = 4π * nrm * sum(@. k * u / (1 - u))
    return est
end

function get_interval(h, ϵ, nrm, β=0.25)
    r_max = Int(ceil(10 / h))
    r_min = 0
    r = 0
    while r_max - r_min >= 2
        r = Int(floor((r_max + r_min) / 2))
        est = err_right(h, r, nrm; β=β)
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
        est = err_left(h, l; β=β)
        if est > ϵ / 2
            l_max = l
        else
            l_min = l
        end
    end
    l = l_min

    return l, r
end

function compute_integral(A, B, V, h, σ, λ_right, l, r, β=0.25)
    n, m = size(A * V)
    X̃ = zeros(ComplexF64, n, m)
    Z = zeros(ComplexF64, n, m)
    k = l:r
    xs = om99_x.(k, h, β=β)
    dxs = om99_dx.(k, h, β=β)
    sinxs = om99_sinx.(k, h, β=β)
    BV = B * V
    for (x, dx, sinx) in zip(xs, dxs, sinxs)
        Z .=
            (im * x * B + A + (σ - λ_right) * B) \ BV .-
            (-im * x * B + A + (σ - λ_right) * B) \ BV
        Z .= (dx * sinx * im / π) * Z
        X̃ .+= Z
    end
    X̃ .= h .* X̃
    return exp(-σ + λ_right) * X̃
end

function eigvals(A::Matrix, B::UniformScaling)
    return LinearAlgebra.eigvals(A / B.λ)
end


"""
A reference implementation of the computation of exp(B⁻¹A)V using the DE formula.
The optional arguments (and their default values) are as follows:

- `B=I`
- `V=I`
- `σ=-0.5`: shift parameter.
- `lr=nothing`: the trunaction points of the DE formula
    - if `lr=nothing`, the truncation points are computed based on Algorithm 1.
- `ϵ=eps()`: tolerance for the truncation error
- `λ_right=nothing`: the right-most eigenvalue of B⁻¹A
    - if `λ_right=nothing` it is computed in the function
- `nrm=nothing`: the norm ∥[B⁻¹A + (σ-λ_right)I]⁻¹∥₂.
    - if nothing, it is computed in the function.
"""
function expm_de(A, h; B=I, V=I, σ=-0.5, lr=nothing, ϵ=eps(), λ_right=nothing, nrm=nothing)
    if isnothing(λ_right)
        λ = eigvals(A, B)
        i = argmax(real(λ))
        λ_right = λ[i]
    end

    if isnothing(nrm)
        nrm = opnorm((A + (σ - λ_right) * B) \ B)
    end

    ϵ̃ = ϵ / abs(exp(λ_right - σ))

    if isnothing(lr)
        l, r = get_interval(h, ϵ̃, nrm)
    else
        l, r = minimum(lr), maximum(lr)
    end
    X = compute_integral(A, B, V, h, σ, λ_right, l, r)
    result = ExpmDEResult(X, σ, [l], [r], [h], [eps()])
    return result
end

function expm_de(
    A; h0=0.2, B=I, V=I, σ=-0.5, ϵ=eps(), λ_right=nothing, nrm=nothing, η=2.0, h_min=1e-3
)
    if isnothing(λ_right)
        λ = eigvals(A, B)
        i = argmax(real(λ))
        λ_right = λ[i]
    end

    if isnothing(nrm)
        nrm = opnorm((A + (σ - λ_right) * B) \ B)
    end

    ϵ̃ = ϵ / abs(exp(λ_right - σ))
    n, m = size(A * V)
    X = zeros(ComplexF64, n, m)
    result = ExpmDEResult(X, σ, zeros(Int, 0), zeros(Int, 0), zeros(0), zeros(0))

    h1, h2, h3 = h0, h0 / 2, h0 / 4
    l1, r1 = get_interval(h1, ϵ̃ / 2, nrm)
    l2, r2 = get_interval(h2, ϵ̃ / 2, nrm)
    l3, r3 = get_interval(h3, ϵ̃ / 2, nrm)
    X1 = compute_integral(A, B, V, h1, σ, λ_right, l1, r1)
    X2 = compute_integral(A, B, V, h2, σ, λ_right, l2, r2)
    X3 = compute_integral(A, B, V, h3, σ, λ_right, l3, r3)

    should_continue = true
    h, l, r = [h1, h2, h3], [l1, l2, l3], [r1, r2, r3]
    errest = fill(Inf, 100)
    max_loop = 5
    loop = 1
    while should_continue && loop < max_loop
        loop += 1
        ϵ1, ϵ2 = opnorm(X1 - X3), opnorm(X2 - X3)


        ρ = h1 * h2 * log(ϵ1 / ϵ2) / (h1 - h2)
        γ = ϵ1 * exp(ρ / h1)

        ϵ3 = γ * exp(-ρ / h3)

        errest[loop] = ϵ1 * abs(exp(λ_right - σ))
        errest[loop+1] = ϵ2 * abs(exp(λ_right - σ))
        errest[loop+2] = ϵ3 * abs(exp(λ_right - σ))

        if ϵ3 < ϵ̃ / η
            X .= X3
            should_continue = false
        else
            h4 = ρ / log(γ / (ϵ̃ / η))
            if 2 * ϵ1 <= ϵ2 || h4 < h_min || !isfinite(h4)
                h1, h2, h3 = h2, h3, h3/2
                X1 .= X2
                X2 .= X3
                l3, r3 = get_interval(h3, ϵ̃ / 2, nrm)
                push!(h, h3); push!(l, l3); push!(r, r3)
                X3 = compute_integral(A, B, V, h3, σ, λ_right, l3, r3)
            else
                l4, r4 = get_interval(h4, ϵ̃ / 2, nrm)
                X4 = compute_integral(A, B, V, h4, σ, λ_right, l4, r4)
                X .= X4
                push!(h, h4)
                push!(l, l4)
                push!(r, r4)
                errest[loop + 3] = ϵ
                should_continue = false
            end
        end
    end

    result.h, result.r, result.l = h, r, l
    result.errest = filter(!isinf, errest)
    result.X .= X
    return result
end
