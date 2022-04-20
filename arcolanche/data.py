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


def dyadic_data(region='africa'):
    """Load raw data.

    Parameters
    ----------
    region : str,'africa'
        'africa', 'middle east', 'asia'
    """

    from datetime import datetime
    
    if region=='africa':
        df = pd.read_csv('%s/%s'%(DATADR,'ACLED-Version-7-All-Africa-1997-2016_csv_dyadic-file.csv'),
                         encoding='latin1')
        df['EVENT_DATE'] = df['EVENT_DATE'].map(lambda t: datetime.strptime(t,'%d/%m/%Y'))
    elif region=='asia':
        df = pd.read_csv('%s/%s'%(DATADR,'Asia_2016-2018_Sept29.csv'))
        df['EVENT_DATE'] = df['EVENT_DATE'].map(lambda t: datetime.strptime(t,'%Y-%m-%d'))
    elif region=='middle east':
        df = pd.read_csv('%s/%s'%(DATADR,'MiddleEast_2016-2018_Sep29-1.csv'))
        df['EVENT_DATE'] = df['EVENT_DATE'].map(lambda t: datetime.strptime(t,'%d-%B-%Y'))
    else:
        raise NotImplementedError

    # Standardize col names
    if 'ASSOC_ACTOR_1' in df.columns:
        df.rename(columns={'ASSOC_ACTOR_1':'ALLY_ACTOR_1'}, inplace=True)
    if 'ASSOC_ACTOR_2' in df.columns:
        df.rename(columns={'ASSOC_ACTOR_2':'ALLY_ACTOR_2'}, inplace=True)
    df['ACTOR1'].fillna('', inplace=True)
    df['ACTOR2'].fillna('', inplace=True)
    df['ALLY_ACTOR_1'].fillna('', inplace=True)
    df['ALLY_ACTOR_2'].fillna('', inplace=True)

    # make all event types lower case since there are inconsistencies in capitalization
    df['EVENT_TYPE']=df['EVENT_TYPE'].map(lambda x:x.lower())

    df['LATITUDE'].values[:] = pd.to_numeric(df['LATITUDE'])
    df['LONGITUDE'].values[:] = pd.to_numeric(df['LONGITUDE'])

    return df


# =================================== #
# Useful quick data access functions. #
# =================================== #
class ACLED2020():
    fname = (f'{DATADR}/1997-01-01-2020-07-23-Eastern_Africa-Middle_Africa-Northern_Africa-'+
             'Southern_Africa-Western_Africa.csv')
    df = pd.read_csv(fname)
    df.event_date = pd.to_datetime(df.event_date)
    df.sort_values('event_date', inplace=True)

    @classmethod
    def battles_df(cls, pre_covid=True):
        """
        Parameters
        ----------
        pre_covid : bool, True

        Returns
        -------
        pd.DataFrame
        """

        if pre_covid:
            ix = ((cls.df.event_date>=pd.to_datetime('1997/01/01')) &
                  (cls.df.event_date<=pd.to_datetime('2019/12/31')))
            df = cls.df.loc[ix]
        else:
            df = cls.df

        return df.loc[df.event_type=='Battles']

    @classmethod
    def vac_df(cls, pre_covid=True):
        """
        Parameters
        ----------
        pre_covid : bool, True

        Returns
        -------
        pd.DataFrame
        """

        if pre_covid:
            ix = ((cls.df.event_date>=pd.to_datetime('1997/01/01')) &
                  (cls.df.event_date<=pd.to_datetime('2019/12/31')))
            df = cls.df.loc[ix]
        else:
            df = cls.df

        return df.loc[df.event_type=='Violence against civilians']

    @classmethod
    def riots_and_protests_df(cls, pre_covid=True):
        """
        Parameters
        ----------
        pre_covid : bool, True

        Returns
        -------
        pd.DataFrame
        """

        if pre_covid:
            ix = ((cls.df.event_date>=pd.to_datetime('1997/01/01')) &
                  (cls.df.event_date<=pd.to_datetime('2019/12/31')))
            df = cls.df.loc[ix]
        else:
            df = cls.df

        ix = (df.event_type=='Riots') | (df.event_type=='Protests')
        return df.loc[ix]
#end ACLED2020
