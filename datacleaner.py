import pandas as pd 
from sklearn.impute import SimpleImputer
import numpy as np
import json

class dataclean:
    def readdata(self):
        with open('filedetail.json','r') as file:
            self.detail=file.read()
        self.filename=json.loads(self.detail)
        self.data=pd.read_csv(self.filename["filename"])
        self.dataarr=self.data.iloc[:,:].values
        #finding the columns with numeric values to fill missing values
        self.datasample=self.dataarr[1]
        self.inum=[]  # list to hold the index of columns with numeric values
        for i in self.datasample:
            if type(i) in [int, float]:
                self.inum.append(np.where(self.datasample==i)[0][0])  #appending the column number to the inum list

    def findemptyindex(self):
        #find if the data has empty cells or missing values
        self.emptyindex = list(map(list,np.where(pd.isna(np.array(self.dataarr)))))
        self.eind=[]
        for i in zip(self.emptyindex[0],self.emptyindex[1]):  #zip is used to combine values from different containers into one entity
            self.eind.append(list(i))
        #print(self.eind)

    def imputing(self):
        #Fill the empty numerical values
        imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
        for imp in self.inum:
            imputer = imputer.fit(self.dataarr[:,imp].reshape(len(self.dataarr[:,imp]),1))
            datafilled = imputer.transform(self.dataarr[:,imp].reshape(len(self.dataarr[:,imp]),1))
            print ("\n" , datafilled )
            self.dataarr[:,imp]=datafilled.reshape(len(self.dataarr[:,imp])) #replace filled data in the numpy arrray
        self.findemptyindex()
        self.missdata=[]
        self.rowlist=[]
        for i in self.eind:
            self.rowlist.append(i[0])
        self.rowlist=np.unique(self.rowlist)
        for i in self.rowlist :
            self.missdata.append(self.dataarr[i,:])
        for i in self.rowlist:
            self.dataarr=np.delete(self.dataarr,i,axis=0)
        np.savetxt('missingdata.csv',self.missdata,delimiter=',',fmt='%s')        
        np.savetxt('datamodified.csv',self.dataarr,delimiter=',',fmt='%s')

    def outliers(self):
        for imp in self.inum:
            stddev=np.nanstd(self.dataarr[:,imp].astype('float32')) #find the standard deviation in the numeric column
            datamean=np.nanmean(self.dataarr[:,imp].astype('float32'))  #find the mean of the column
            cutoff=stddev*2 #set cutoff to 3 time the std deviation
            lowlimit=datamean - cutoff  #lower limit
            uplimit=datamean + cutoff  #upper limit
            self.outdata=[]  # seperated outlier data
            print("stddev",stddev,"mean",datamean,"lowlimit",lowlimit,"uplimit",uplimit)
            for i in self.dataarr[:,imp]:
                if i < lowlimit or i > uplimit:
                    loc=np.where(self.dataarr==i)[0][0]   #find the location index
                    self.outdata.append(self.dataarr[loc,:])  #append the row to the outlier data
                    self.dataarr=np.delete(self.dataarr,loc,axis=0)
            self.outdata=np.array(self.outdata)
            np.savetxt("outliers.csv",self.outdata,delimiter=',',fmt='%s')



if __name__ == "__main__":
    dc=dataclean()  #create an object of class dataclean
    dc.readdata()   #read the data and identify the numerical columns
    dc.outliers()   #find the outliers and seperate them, must be done before replacing missing values
    dc.imputing()   #find the missing cells and fill numeric, seperate the categorical and save to seperate file
    