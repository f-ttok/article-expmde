using NLsolve

# http://www.cas.cmc.osaka-u.ac.jp/~paoon/Lectures/2018-7Semester-AppliedMath9/04-nlsolve/
function nls(func, params...; ini=[0.0])
    if typeof(ini) <: Number
        r = nlsolve((vout, vin) -> vout[1] = func(vin[1], params...), [ini])
        v = r.zero[1]
    else
        r = nlsolve((vout, vin) -> vout .= func(vin, params...), ini)
        v = r.zero
    end
    return v, r.f_converged
end

function dw15_z(θ, m, α, μ, σ, ν)
    if abs(θ)^3 > eps()
        ζ = -σ + μ * θ * cot(α * θ) + im * ν * θ
    else
        ζ = -σ + μ / α
    end
    return m * ζ
end


function dw15_dz(θ, m, α, μ, ν)
    if abs(θ)^4 > eps()
        αθ = α * θ
        dζ = μ * (cot(αθ) - αθ * csc(αθ)^2) + im * ν
    else
        dζ = -2 * θ * α * μ / 3 + -4 * α^3 * μ * θ^3 / 45 + im * ν
    end
    return m * dζ
end

function compute_integral(A, m, α, μ, σ, ν; B=I, V=I)
    X = zero(A) * V * im
    BV = B * V
    θ_list = [-π + (2k - 1) * π / m for k = 1:m]
    z_list = dw15_z.(θ_list, m, α, μ, σ, ν)
    dz_list = dw15_dz.(θ_list, m, α, μ, ν)
    for (z, dz) in zip(z_list, dz_list)
        X += dz * exp(z) * ((z * B - A) \ BV)
    end
    return X / (im * m)
end

function fz(c, param)
    α, m, k = param
    z = c
    z += log(eps() / k) / m
    z += -2 * α * c^2 * (sin(α * π))^2 / (2 * α * c^2 * (sin(α * π))^2 - π * sin(2 * α * π) * sinh(α * c)^2)
    z += (2 * c * sin(α * π)^2 * sinh(α * c)^2) / (α * (2 * α * c^2 * (sin(α * π))^2 - π * sin(2 * α * π) * sinh(α * c)^2))
    return z
end

mutable struct ExpmTalbotResult
    X
    m_stepsize
    estimates
    errors
end

function expm_talbot(A, m_stepsize, m_max, ϵ; B=I, V=I, Ref=nothing)
    estimates = Dict()
    errors = Dict()
    m = 4

    n = size(A, 1)
    X = zeros(ComplexF64, n, n)
    X_old = zeros(ComplexF64, n, n)

    while m <= m_max
        α, μ, σ, ν = 0.6407, 0.5017, 0.6122, 0.2645
        X = compute_integral(A, m, α, μ, σ, ν, B=B, V=V)
        estimates[m] = opnorm(X - X_old)
        if !isnothing(Ref)
            errors[m] = opnorm(X - Ref)
        end
        X_old .= X
        if (
            m >= 24
            && estimates[m] > estimates[m-m_stepsize]
            && estimates[m] > estimates[m-2*m_stepsize]
            && estimates[m-2*m_stepsize] < 1e-2
        )
            break
        end
        m += m_stepsize
    end

    k = eps() * exp(1.529 * (m - m_stepsize))
    c = 1.0
    α = 0.6407
    while m <= m_max
        c = nls(fz, (α, m, k), ini=c)[1]
        b = c * (sin(α * π))^2 / (2 * α * c^2 * (sin(α * π))^2 - π * sin(2 * π * α) * (sinh(α * c))^2)
        σ = 2 * α * c^2 * b
        μ = 2 * (sinh(α * c))^2 * b
        ν = (sinh(2 * α * c) - 2 * α * c) * b
        X = compute_integral(A, m, α, μ, σ, ν, B=B, V=V)

        estimates[m] = opnorm(X - X_old)
        if !isnothing(Ref)
            errors[m] = opnorm(X - Ref)
        end
        X_old .= X
        if estimates[m] < ϵ
            break
        end
        m += m_stepsize
    end

    result = ExpmTalbotResult(X, m_stepsize, estimates, errors)
end