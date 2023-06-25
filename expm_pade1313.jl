using LinearAlgebra
using OffsetArrays

function expm_pade1313(A::AbstractMatrix)
    n = size(A, 1)
    m = 13
    b = OffsetArray([64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1], 0:m)
    θ13 = 5.371920351148152

    μ = tr(A) / n
    A = A - μ*I
    s = max(Int(floor(log2(opnorm(A, 1) / θ13))), 0)
    A = A / (2^s)
    A2 = A^2
    A4 = A2^2
    A6 = A2 * A4
    U = A * (A6*(b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I)
    V = A6 * (b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    R = (-U + V) \ (U + V)
    return exp(μ) * R^(2^s)
end
