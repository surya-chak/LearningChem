# using DiffEqFlux
# using DiffEqSensitivity
# using LinearAlgebra
# using Flux

# # using OrdinaryDiffEq
# using DifferentialEquations
# using Optim
# using Zygote
# using PyCall
# using PyPlot

using Flux
using DiffEqFlux
using DifferentialEquations
using Optim
using DiffEqSensitivity
using Zygote

using PyCall
using PyPlot
using DelimitedFiles
using LinearAlgebra
using DataInterpolations
using Dierckx
using JLD2
using FileIO


include("IOUtils.jl")

NSyst=1;
NInst=1;

# =================================
# Reading in the system information
# =================================
# reading in the data
data=systIO(NSyst,NInst)

# unnpacking the data read in 
XDatLong=Float64.(data[1]);
UDatLong=Float64.(data[2]);
TVecLong=Float64.(data[3]);
YVecLong=Float64.(data[4]);
TargetVecLong=Float64.(data[5]);


iShort=9;
XDat=XDatLong[1:iShort,:];
UDat=UDatLong[1:iShort,:];
TVec=TVecLong[1:iShort,:];
YVec=YVecLong[1:iShort,:];
TargetVec=TargetVecLong[1:iShort,:];

TVec=TVecLong[1:iShort];

nX=size(XDat,2);
nU=size(UDat,2);
nn=size(TVec,1);
TFin=TVec[end];

# finding out the control time till which control is applied
for nT in 1:1:nn
    if UDat[nT,1]==0.0
        global NUon=nT-1;
        break;
    end
end
TCtrlOn=TVec[NUon]
UVec=Float64.(UDat[1,:]);
U=zeros(nU);

# =====================================
# Defining the parameters of the system
# =====================================

Linear=0.01*rand(nX,nX);
Quadratic=0.001*rand(nX,nX,nX);

BMat=0.01*rand(nX,nU);

# packing up the coefficients into parameters vector
pLin=Linear[:];
len_pL=length(pLin);

pQuad=Quadratic[:];
len_pQ=length(pQuad);

pB=BMat[:];
len_pB=length(pB);

p=[pLin;pQuad;pB];

# RHS of the system
function Syst_RHS!(dX,X,p,t)
    # Dismantling the parameters of the neural network
    LTerm=reshape(p[1:len_pL],nX,nX);
    QTerm=reshape(p[len_pL+1:len_pQ+len_pL],nX,nX,nX);
    BTerm=reshape(p[len_pL+len_pQ+1:end],nX,nU);

    U=zeros(nU)

    if t<=TCtrlOn
        U=UVec;
    else
        U.=0.0;
    end
    for iState=1:1:nX
        dX[iState]=dot(LTerm[iState,:],X)+X'*(transpose(QTerm[iState,:,:])*X);
        dX[iState]=dX[iState]+dot(BTerm[iState,:],U); # Adding the control
    end
end

# ==================
# Setup ODE problem
# ==================
X0=XDat[1,:]; #init conditions
println("size X0 is", size(X0))
TSpan=(0.0,TFin);
prob_nn = ODEProblem(Syst_RHS!, X0, TSpan, p);

println("Solving with untrained params...")
sol = Array(solve(prob_nn, Tsit5(),saveat=TVec,reltol=1e-2))
# ================
# Training set up
# ================
# Forward pass function
function predict_adjoint() # Trainable layer
    Array(solve(prob_nn, Tsit5(), saveat=TVec, reltol=1e-2))
end

# Loss function
function loss_adjoint()
    prediction = predict_adjoint()
    loss = sum((prediction - XDat').^2); # L2 norm
    return loss
end

# Defining learning parameters
opt=ADAM(0.1);
params=Flux.params(p)

losshistory = []
cb = function () #callback function to observe training
    push!(losshistory,loss_adjoint());
    display(loss_adjoint());
end

# Display the ODE with the initial parameter values.
cb()
@info "Start training"
Flux.train!(loss_adjoint, params, Iterators.repeated((), 100), opt, cb = cb)
@info "Finished Training"

 
