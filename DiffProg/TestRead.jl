using DelimitedFiles



# Reading from the publicData
nInst=20;
nSys=12;


path="./../public_data/"
fname="system_01_instance_00.csv"

data=readdlm(string(path,fname),',',header=true)
system_data=data[1][:,3:end]
system_data[system_data.==""].=0.0;

