import pandas as pd 
from sklearn.impute import SimpleImputer
import numpy as np

def imputing():
    data=pd.read_csv("Data.csv")
    dataarr=data.iloc[:,:].values
    print(dataarr)
    #finding the columns with numeric values to fill missing values
    datasample=dataarr[1]
    inum=[]  # list to hold the index of columns with numeric values
    for i in datasample:
        if type(i) in [int, float]:
            inum.append(np.where(datasample==i)[0][0])  #appending the column number to the inum list
    #find if the data has empty cells or missing values
    emptyindex = list(map(list,np.where(pd.isna(np.array(dataarr)))))
    eind=[]
    for i in zip(emptyindex[0],emptyindex[1]):  #zip is used to combine values from different containers into one entity
        eind.append(list(i))
    imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
    for imp in inum:
        imputer = imputer.fit(dataarr[:,imp].reshape(len(dataarr[:,imp]),1))
        datafilled = imputer.transform(dataarr[:,imp].reshape(len(dataarr[:,imp]),1))
        print ("\n" , datafilled )
        dataarr[:,imp]=datafilled.reshape(len(dataarr[:,imp])) #replace filled data in the numpy arrray
    print(dataarr)
    datamod=pd.DataFrame(dataarr)  #converting to dataframe
    np.savetxt('datamodified.csv',dataarr,delimiter=',',fmt='%s')

def outliers():
    data=pd.read_csv("Data.csv")
    dataarr=data.iloc[:,:].values
    #finding the columns with numeric values to fill missing values
    datasample=dataarr[1]
    inum=[]  # list to hold the index of columns with numeric values
    for i in datasample:
        if type(i) in [int, float]:
            inum.append(np.where(datasample==i)[0][0])  #appending the column number to the inum list
    print("inum",inum)
    for imp in inum:
        print(dataarr[:,imp])
        stddev=np.nanstd(dataarr[:,imp].astype('float32')) #find the standard deviation in the numeric column
        datamean=np.nanmean(dataarr[:,imp].astype('float32'))  #find the mean of the column
        cutoff=stddev*2 #set cutoff to 3 time the std deviation
        lowlimit=datamean - cutoff  #lower limit
        uplimit=datamean + cutoff  #upper limit
        outdata=[]  # seperated outlier data
        print("stddev",stddev,"mean",datamean,"lowlimit",lowlimit,"uplimit",uplimit)
        for i in dataarr[:,imp]:
            if i < lowlimit or i > uplimit:
                loc=np.where(dataarr==i)[0][0]   #find the location index
                print("loc",loc,'data',i)
                outdata.append(dataarr[loc,:])  #append the row to the outlier data
                dataarr=np.delete(dataarr,loc,axis=0)
        outdata=np.array(outdata)
        print("outdata",outdata,"\n","data",'\n',dataarr)



if __name__ == "__main__":
    #imputing()  
    outliers()