using DelimitedFiles

function systIO(NSyst,NInst)

    path="./../public_data/"
    fname=string("system_",lpad(NSyst,2,"0"),"_instance_",lpad(NInst,2,"0"),".csv")
    # fname="system_01_instance_00.csv"

    data=readdlm(string(path,fname),',',header=true)
    system_data=data[1][:,3:end]

    # making empty spaces=0
    system_data[system_data.==""].=0.0;
    headers=data[2][3:end]


    TVec=system_data[:,1];          # Getting time vector
    YVec=system_data[:,2];          # Getting time vector
    TargetVec=system_data[:,3];          # Getting time vector

    # ===========
    # Get X data
    # ===========

    # count to the start of X
    for varCtr in 1:1:length(headers)
        if headers[varCtr] == "X1"
            global XSt=varCtr;
            break;
        end
    end

    ctr=0;
    # count to total number of X
    for varCtr in 1:1:length(headers)
        if headers[varCtr+XSt-1] == string("X",varCtr)
            ctr+=1;
        else
            break
        end
    end
    XEnd=ctr+XSt-1;
    NX=XEnd-XSt+1;

    XDat=system_data[:,XSt:XEnd];     # Final matrix containing the X data


    # ===========
    # Get Control data
    # ===========

    # count to the start of U
    USt=XEnd+1;
    # count to total number of U inputs
    UEnd=length(headers);
    NU=UEnd-USt+1;
    UDat=system_data[:,USt:UEnd];     # Final matrix containing the U data

    return XDat,UDat,TVec,YVec,TargetVec
    
end
