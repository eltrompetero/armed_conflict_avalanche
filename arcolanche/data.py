# ====================================================================================== #
# Module for importing data from ACLED.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import sys

from .utils import *

# try to find data directory either in cwd or above
DATADR = os.getcwd()+'/../data'
if not os.path.isdir(DATADR):
    DATADR = os.getcwd()+'/data'
assert os.path.isdir(DATADR), DATADR



# =================================== #
# Useful quick data access functions. #
# =================================== #
class ACLED2020():
    fname = (f'{DATADR}/ACLED_data.csv')
    if os.path.isfile(fname):
        df = pd.read_csv(fname)
        df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'])
        df.sort_values('EVENT_DATE', inplace=True)

    @classmethod
    def battles_df(cls, pre_covid=True, to_lower=False, year_range=False):
        """
        Parameters
        ----------
        pre_covid : bool, True
        to_lower : bool, False
            If True, rename all cols to lower case.
        year_range : tuple , False
            If a tuple is passed, the first element of the tuple is taken
            to be the lower cutoff of year and second element is upper
            cutoff.

        Returns
        -------
        pd.DataFrame
        """

        if(year_range):
            ix = ((cls.df['EVENT_DATE']>=pd.to_datetime(f'{year_range[0]}/01/01')) &
                  (cls.df['EVENT_DATE']<=pd.to_datetime(f'{year_range[1]}/12/31')))
            df = cls.df.loc[ix]
        else:
            if pre_covid:
                ix = ((cls.df['EVENT_DATE']>=pd.to_datetime('1997/01/01')) &
                    (cls.df['EVENT_DATE']<=pd.to_datetime('2019/12/31')))
                df = cls.df.loc[ix]
            else:
                df = cls.df
        
        if to_lower:
            df.columns = [i.lower() for i in df.columns]
            return df.loc[df['event_type']=='Battles']
        return df.loc[df['EVENT_TYPE']=='Battles']

    @classmethod
    def vac_df(cls, pre_covid=True, to_lower=False, year_range=False):
        """
        Parameters
        ----------
        pre_covid : bool, True
        to_lower : bool, False
            If True, rename all cols to lower case.
        year_range : tuple , False
            If a tuple is passed, the first element of the tuple is taken
            to be the lower cutoff of year and second element is upper
            cutoff.

        Returns
        -------
        pd.DataFrame
        """

        if(year_range):
            ix = ((cls.df['EVENT_DATE']>=pd.to_datetime(f'{year_range[0]}/01/01')) &
                  (cls.df['EVENT_DATE']<=pd.to_datetime(f'{year_range[1]}/12/31')))
            df = cls.df.loc[ix]
        else:
            if pre_covid:
                ix = ((cls.df['EVENT_DATE']>=pd.to_datetime('1997/01/01')) &
                    (cls.df['EVENT_DATE']<=pd.to_datetime('2019/12/31')))
                df = cls.df.loc[ix]
            else:
                df = cls.df

        if to_lower:
            df.columns = [i.lower() for i in df.columns]
            return df.loc[df['event_type']=='Violence against civilians']
        return df.loc[df['EVENT_TYPE']=='Violence against civilians']

    @classmethod
    def riots_and_protests_df(cls, pre_covid=True, to_lower=False, year_range=False):
        """
        Parameters
        ----------
        pre_covid : bool, True
        to_lower : bool, False
            If True, rename all cols to lower case.
        year_range : tuple , False
            If a tuple is passed, the first element of the tuple is taken
            to be the lower cutoff of year and second element is upper
            cutoff.

        Returns
        -------
        pd.DataFrame
        """

        if(year_range):
            ix = ((cls.df['EVENT_DATE']>=pd.to_datetime(f'{year_range[0]}/01/01')) &
                  (cls.df['EVENT_DATE']<=pd.to_datetime(f'{year_range[1]}/12/31')))
            df = cls.df.loc[ix]
        else:
            if pre_covid:
                ix = ((cls.df['EVENT_DATE']>=pd.to_datetime('1997/01/01')) &
                    (cls.df['EVENT_DATE']<=pd.to_datetime('2019/12/31')))
                df = cls.df.loc[ix]
            else:
                df = cls.df

        if to_lower:
            df.columns = [i.lower() for i in df.columns]
            return df.loc[(df['event_type']=='Riots') | (df['event_type']=='Protests')]
        ix = (df['EVENT_TYPE']=='Riots') | (df['EVENT_TYPE']=='Protests')
        return df.loc[ix]
#end ACLED2020
