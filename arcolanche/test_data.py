# ====================================================================================== #
# Module for importing data from ACLED.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from .data import *



def test_ACLED2020():
    assert ACLED2020.df.shape==(215659,31)
    assert ACLED2020.battles_df().shape==(55367,31)
    assert ACLED2020.vac_df().shape==(50348,31)
    assert ACLED2020.riots_and_protests_df().shape==(63108,31)

#def test_tpixelate():
#    df=pd.read_csv('sample_data_1.txt')
#    for i in range(len(df)):
#        df.iloc[i]=pd.to_datetime(df.iloc[i].values[0])
#    splitdf=tpixelate(df, 1, df.iloc[0].values[0], df.iloc[-1].values[0])
#    assert len(splitdf)==1 and len(splitdf[0])==4
#
#    df=pd.read_csv('sample_data_2.txt')
#    for i in range(len(df)):
#        df.iloc[i]=pd.to_datetime(df.iloc[i].values[0])
#    splitdf=tpixelate(df, 1, df.iloc[0].values[0], df.iloc[-1].values[0])
#    assert len(splitdf)==4
#
#    df=pd.read_csv('sample_data_3.txt')
#    for i in range(len(df)):
#        df.iloc[i]=pd.to_datetime(df.iloc[i].values[0])
#    splitdf=tpixelate(df, 1, df.iloc[0].values[0], df.iloc[-1].values[0])
#    assert len(splitdf[0])==1 and len(splitdf[1])==2 and len(splitdf[2])==1
#
#if __name__=='__main__':
#    test_tpixelate()
