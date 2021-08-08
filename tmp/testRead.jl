# using JLD2
using HDF5

d=h5open("init_params_torch_system01.h5","r");

B=read(d["Input_weights"])

L=read(d["Linear_weights"])

Q=read(d["Quadratic_weights"])





