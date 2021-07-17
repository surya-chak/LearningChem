using DelimitedFiles



# Reading from the publicData
nInst=20;
nSys=12;


path="./../public_data/"
fname="system_01_instance_00.csv"

data=readdlm(string(path,fname),',',header=true)
system_data=data[1][:,3:end]
system_data[system_data.==""].=0.0;
headers=data[2]

# count number of inputs


for varCtr in 1:1:length(headers)
    if headers[varCtr] == "X1"
        XSt=varCtr;
        break;
    end
end

ctr=0;
for varCtr in 1:1:length(headers)
    if headers[varCtr+XSt-1] == string("X",varCtr)
        ctr+=1;
    else
        break
    end

end

USt=
