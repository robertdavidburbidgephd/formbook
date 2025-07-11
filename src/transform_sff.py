import pandas as pd
import numpy as np
import datetime as dt


d24 = pd.read_csv('data/extracted/SFF_2024.psv', sep='|', header=0
# , nrows=1000
, usecols=[ 'Id'
    , 'Course'
    , 'RaceDate'
    , 'RaceTime'
    , 'Race'
    , 'Type'
    , 'Class'
    , 'AgeLimit'
    , 'Prize'
    , 'Ran'
    , 'Distance'    # text, for convenience
    , 'Yards'       # numeric
    , 'Going'
    , 'Limit'
    # , 'WinTime' # redundant
    , 'Seconds'
    , 'FPos'    # derive DNF ; DNF: FPos <- 99
    , 'DstBtn'  # parse to numeric (lengths) ; handle missing
    , 'TotalBtn'    # handle 'Null' and missing
    , 'CardNo'
    , 'HorseName'
    , 'Draw'
    , 'Sp'  # handle missing (if this is needed)
    , 'Age'
    , 'Stone'       # check WeightLBS
    , 'Lbs'         # then drop1
    , 'WeightLBS'   # Stone, Lbs
    , 'Favs'    # canonicalize (if this is needed)
    , 'Aid'     # canonicalize (see note in RP print cards on GB vs IRE visors, etc.)
    , 'Trainer'
    , 'Jockey'
    , 'Allow'
    , 'OR'
    , 'Comments'
    ]
, dtype={ 'Id': np.int32
    , 'Course': pd.StringDtype()
    # , 'RaceDate':  dt.date  ??
    # , 'RaceTime':  dt.time()  ??
    , 'Race': pd.StringDtype()
    , 'Type': pd.StringDtype()
    # , 'Class': np.int8  # has NAs
    , 'AgeLimit': pd.StringDtype()
    , 'Prize': np.int32
    , 'Ran': np.int8
    , 'Distance': pd.StringDtype()
    , 'Yards': np.int16
    , 'Going': pd.StringDtype()
    # , 'Limit': np.int16  # has NAs
    # , 'WinTime': pd.StringDtype()
    , 'Seconds': np.float64
    , 'FPos': pd.StringDtype()
    , 'DstBtn': pd.StringDtype()
    , 'TotalBtn': pd.StringDtype()
    , 'CardNo': np.int8
    , 'HorseName': pd.StringDtype()
    # , 'Draw': np.int8   # has NAs
    , 'Sp': np.float64
    , 'Age': np.int8
    , 'Stone': np.int8
    , 'Lbs': np.int8
    , 'WeightLBS': np.int8
    , 'Fav': pd.StringDtype()
    , 'Aid': pd.StringDtype()
    , 'Trainer': pd.StringDtype()
    , 'Jockey': pd.StringDtype()
    # , 'Allow': np.int8   # has NAs
    # , 'OR': np.int8   # has NAs
    , 'Comments': pd.StringDtype()
    }
, parse_dates=[2, 3]    # works but throws a warning for RaceDate as 'd-Mon-yy'
, dayfirst=True
, thousands=','
, encoding='cp1252').\
    rename( { 'Id': 'RaceId'
            , 'Course': 'CourseName'
            , 'Seconds': 'RaceWinTime'}, axis=1 )

print(d24.head())
print(d24.tail())

def fColumns(df):
    print(pd.DataFrame.from_dict({ 'column': df.columns.values, 
                            'dtype': df.dtypes.values,
                            'nmissing': df.isna().apply(sum).values}))
    return


fColumns(d24)

CourseName_ = d24['CourseName'].unique()
CourseName_ = CourseName_[CourseName_.argsort()]
print(CourseName_)

Territory_ = [ 'NI' if x.endswith(' (NI)') else 'IRE' if x.endswith(' (IRE)') else 'GB' for x in CourseName_ ]
def fCourseName(x):
    return x.removesuffix(' (NI)').removesuffix(' (IRE)')


CourseName_ = [ fCourseName(x) for x in CourseName_ ]
d24['CourseName'] = d24['CourseName'].apply(fCourseName)
CourseID_ = [ '{:02d}'.format(n) for n in range(1,len(CourseName_)+1) ]

CourseTable = pd.DataFrame.from_dict({ 'CourseName': CourseName_
                                     , 'CourseID': CourseID_
                                     , 'Territory': Territory_ })

""" https://stackoverflow.com/questions/61330414/pandas-astype-with-date-or-datetime/75990548#75990548
"""
d24['RaceDate'] = d24['RaceDate'].dt.date
d24['RaceTime'] = d24['RaceTime'].dt.time

d24['Md'] = d24.Race.str.contains('Maiden')
d24['Nv'] = d24.Race.str.contains('Novice')
d24['Hc'] = d24.Race.str.contains('Handicap')

d24['F'] = d24.Race.str.contains('Fillies')
d24['M'] = d24.Race.str.contains('Mares')
d24['CG'] = d24.Race.str.contains('Colts & Geldings') | d24.Race.str.contains('Colts And Geldings')
d24['C'] = d24.Race.str.contains('\(Colts\)')
d24['CF'] = d24.Race.str.contains('Colts & Fillies') | d24.Race.str.contains('Colts And Fillies')
d24['NG'] = d24.Race.str.contains('No Geldings')

d24['H'] = ~d24.Type.isna() & (d24.Type == 'h')
d24['Ch'] = ~d24.Type.isna() & (d24.Type == 'c')
d24['NHF'] = ~d24.Type.isna() & (d24.Type == 'b')
d24['Flat'] = d24.Type.isna()
# d24.drop('Type', axis=1, inplace=True)

# Class 47 Dun 14/02/24 ...
d24.loc[d24.Class==47, 'Class'] = np.nan

d24['AgeMin'] = [ int(x[:x.find('Y')]) for x in d24.AgeLimit ]  # int
d24['AgeMax'] = [ int(x[:x.find('Y')]) if x.endswith('only') 
                 else int(x[x.rfind('Y')-1]) if x.find('to') > 0    # < 10
                 else np.nan for x in d24.AgeLimit ]    # float
# d24.drop('AgeLimit', axis=1, inplace=True)

Going_ = d24['Going'].unique()
Going_ = Going_[Going_.argsort()]
print(Going_)

# AW False -> Turf for GB/IRE
d24['AW'] = d24.Going.str.contains('AW')

def fGoing(x):
    return x.replace('AW','').lower().replace('-','').replace('sand','').\
            replace('tapeta','').replace('polytrack','').\
                replace(' in ','').replace(' on ','').replace('places','').replace('(str)','').\
                replace('(','|').replace(')','').replace(' ','').\
                replace('goyieldingodtoyielding','yielding').\
                replace('standard/slow','goodtosoft').\
                replace('standard','good').\
                replace('goodtioyielding','goodtoyielding').\
                replace('goodtofim','goodtofirm')
                

Going__ = [ fGoing(x) for x in Going_ ]

# firm 1.0
# goodtofirm 2.0
# good 3.0
## goodtoyielding 3.5
# goodtosoft, yielding 4.0
## yieldingtosoft 4.5
# soft 5.0
## softtoheavy 5.5
# heavy 6.0

def fGoingNum(x):
    y = x.replace('goodtofirm','2.0').\
            replace('goodtoyielding','3.5').\
            replace('goodtosoft','4.0').\
            replace('yieldingtosoft','4.5').\
            replace('softtoheavy','5.5').\
            replace('firm','1.0').\
            replace('good','3.0').\
            replace('yielding','4.0').\
            replace('soft','5.0').\
            replace('heavy','6.0').partition('|')
    return float(y[0]) if y[2] == '' else (2.0 * float(y[0]) + float(y[2])) / 3.0


d24['GoingNum'] = d24['Going'].apply(fGoing).apply(fGoingNum)

# Limit ...