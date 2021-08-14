# Neural GPROM learning both C and L terms
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
    

## Setup ODE to optimize ##
global ngal=6;

# Number of snapshots
nsnap=300;

# time step between two snapshots
dtsnap=0.005;

#Ratio of dtsnap for the number of time steps for integration of POD ROM
dtsnap_by_dtROM=10;

#total time of integration
time_integre = 1.5-dtsnap;            # use long time integration to see if its stable

#Reynolds Number
rey=1500;



# =======================
# Load coeffs from file
# =======================
path=string("data/m",ngal,"/")

Acoef=readdlm(string(path,"ConstCoeff.txt")); # Constant

Bcoef=readdlm(string(path,"LinearCoeff.txt")); # Linear

CcoefS=readdlm(string(path,"QuadraticCoeff.txt")); # Quadratic

Bcoef=reshape(Bcoef,ngal,ngal);

Quadratic=zeros(ngal,ngal,ngal);

println("filling quadratic term..")

l=0
for k =1:ngal
    for j =1:ngal
        for i=1:ngal
            global l=l+1
            Quadratic[i,j,k]=CcoefS[l,1]
        end
    end
end

avectp=readdlm(string(path,"aVecTp.txt"),',');
nt=size(avectp,1);
println("avectp size", size(avectp))

#define training data
traindata = transpose(avectp);

# ================
### untrained ROM
# ================
constant = Acoef; # this is the constant term in GP
global const linear=Bcoef;
global const quadratic = Quadratic;
daROM=zeros(6,1);

println("constant size:", size(constant))
println("linear size:", size(linear))
println("quad size:", size(quadratic))
println("traindata size:", size(traindata))

# ===================================
# Declaring the trainable parameters:
# ===================================
C_Model = constant[:];
L_Model = linear[:];

len_p1 = size(C_Model,1);
len_p2 = size(L_Model,1);

p_init=[C_Model;L_Model];
p = [C_Model;L_Model];

params=Flux.params(p)
# data=Iterators.repeated((),100)
# ==================================
# RHS of the Galerkin Projection ODE 
# ==================================
function NeuralGP!(daROM, a, p, t)
    p1 = @view p[1:len_p1];
    p2 = @view p[(len_p1+1):end];
    Cterm = p1;
    Lterm = reshape(p2,ngal,ngal);
    for i=1:ngal
        daROM[i] = Cterm[i]+dot(Lterm[i,:],a)+a'*(transpose(quadratic[i,:,:])*a)
    end
end

# ==================
# Setup ODE problem
# ==================
a0=avectp[1,:]; #init conditions
println("size a0 is", size(a0))
tspan=(0.0,time_integre);
prob_nn = ODEProblem(NeuralGP!, a0, tspan, p);

println("Solving with untrained params...")
sol = Array(solve(prob_nn, Tsit5(),saveat=0.0:dtsnap:time_integre))

# ========
## Train
# ========
# Forward pass function
function predict_adjoint() # Trainable layer
    Array(solve(prob_nn, Tsit5(), saveat=dtsnap, reltol=1e-4))
end

# Loss function
function loss_adjoint()
    prediction = predict_adjoint()
    loss = sum((prediction - traindata).^2); # L2 norm
    return loss
end

opt=ADAM(0.1);


losshistory = []
CpclHistory = []
EigenHistory = []

cb = function () #callback function to observe training
    l=loss_adjoint()
    Cpcl=sum((p-p_init)[:].^2)

    p2 = @view p[(len_p1+1):end];
    Lterm = reshape(p2,ngal,ngal);
    EigenValue=eigvals(Lterm);
    
    println("Loss Value= $l , CpclValue=$Cpcl ")
    push!(losshistory,l);
    push!(CpclHistory,Cpcl);
    push!(EigenHistory,EigenValue);
end

# Display the ODE with the initial parameter values.
cb()
@info "Start training"
Flux.train!(loss_adjoint, params, Iterators.repeated((), 4000), opt, cb = cb)
@info "Finished Training"


# ================================
# saving the trained coefficients
# ================================

outPath=string("TrainedData/m",ngal,"/")


# saving the coefficient matrices
p1 = @view p[1:len_p1];
p2 = @view p[(len_p1+1):end];
Ctrained = p1;
Ltrained = reshape(p2,ngal,ngal);

# Writing out the trained constant coefficients
writedlm(string(outPath,"ConstCoeff.txt"),Ctrained); # Constant

# Writing out the trained Linear coefficients
writedlm(string(outPath,"Linearcoeff.txt"),Ltrained,','); # Constant

# Writing out the loss history
writedlm(string(outPath,"lossHistory.txt"),losshistory); # Constant

# Saving out the convergence histories as a jld2 file:

jldsave(string(outPath,"trainingHistories.jld2");CpclHistory,EigenHistory,losshistory)



