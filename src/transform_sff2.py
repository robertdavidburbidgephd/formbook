import pandas as pd
import numpy as np
import datetime as dt
import re

files = [ 'SFF_2016-2019.psv', 'SFF_2020-2023.psv', 'SFF_2024.psv' ]
fpath = 'data/prep/'

def fColumns(df):
    print(pd.DataFrame.from_dict({ 'column': df.columns.values, 
                            'dtype': df.dtypes.values,
                            'nmissing': df.isna().apply(sum).values}))
    return


def fTerritory(x):
    y = re.search(r' \(([A-Z]+)\)', x)
    return y.group(1) if y else 'GB'


def fCourseName(x):
    return x.removesuffix(' (NI)').removesuffix(' (IRE)')


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


dfl = dict()
for f in files:
    # f = files[1]
    dfl[f] = pd.read_csv(fpath + f, sep='|', header=0
        # , nrows=1000
        , usecols=[ 'Id', 'Course', 'RaceDate', 'RaceTime', 'Race'
                    , 'Type', 'Class', 'AgeLimit', 'Prize', 'Ran'
                    , 'Yards', 'Going'
        , 'Limit'
        , 'Seconds'
        , 'FPos'    # derive DNF ; DNF: FPos <- 99
        , 'DstBtn'  # parse to numeric (lengths) ; handle missing
        , 'TotalBtn'    # handle 'Null' and missing
        , 'CardNo'      # missing -> not GB/IRE
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
        , 'OR'  # contains '' and '-' as missing
        , 'Comments'
        ]
    , dtype={ 'Id': np.int32, 'Course': pd.StringDtype(), 'Race': pd.StringDtype()
            , 'Type': pd.StringDtype(), 'AgeLimit': pd.StringDtype()
            , 'Prize': np.float64, 'Ran': np.int8, 'Yards': np.int16
            , 'Going': pd.StringDtype()
        , 'Seconds': np.float64
        , 'FPos': pd.StringDtype()
        , 'DstBtn': pd.StringDtype()
        , 'TotalBtn': pd.StringDtype()
        # , 'CardNo': np.int8   # has NAs
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
        , 'OR': pd.StringDtype()
        , 'Comments': pd.StringDtype()
        }
    , parse_dates=[2, 3]    # works but throws a warning for RaceDate as 'd-Mon-yy'
    , dayfirst=True
    , thousands=','
    , encoding='cp1252').\
        rename( { 'Id': 'RaceId'
                , 'Course': 'CourseName'
                , 'Seconds': 'RaceWinTime'}, axis=1 )
    # print(dfl[f].head())
    # print(dfl[f].tail())
    # fColumns(dfl[f])


df = pd.concat(dfl.values(), axis=0).reset_index(drop=True)

# remove non-GB/IRE races [not in 2024 data]
df['Territory'] = df.CourseName.apply(fTerritory)
df.loc[df.Territory=='NI', 'Territory'] = 'IRE'
df = df[(df.Territory=='GB') | (df.Territory=='IRE')]
df['CourseName'] = df['CourseName'].apply(fCourseName)

""" https://stackoverflow.com/questions/61330414/pandas-astype-with-date-or-datetime/75990548#75990548
"""
df['RaceDate'] = df['RaceDate'].dt.date
df['RaceTime'] = df['RaceTime'].dt.time

# df[['RaceId','CourseName','Territory','RaceDate','RaceTime']]

df['Md'] = df.Race.str.contains('Maiden')
df['Nv'] = df.Race.str.contains('Novice')
df['Hc'] = df.Race.str.contains('Handicap')

df['F'] = df.Race.str.contains('Fillies')
df['M'] = df.Race.str.contains('Mares')
df['CG'] = df.Race.str.contains('Colts & Geldings') | df.Race.str.contains('Colts And Geldings')
df['C'] = df.Race.str.contains('\(Colts\)')
df['CF'] = df.Race.str.contains('Colts & Fillies') | df.Race.str.contains('Colts And Fillies')
df['NG'] = df.Race.str.contains('No Geldings')

# df[['RaceId','Race','Md','Nv','Hc','F','M','CG','C','CF','NG']]

df['H'] = ~df.Type.isna() & (df.Type == 'h')
df['Ch'] = ~df.Type.isna() & (df.Type == 'c')
df['NHF'] = ~df.Type.isna() & (df.Type == 'b')
df['Flat'] = df.Type.isna()

# df[['RaceId','Type','H','Ch','NHF','Flat']]
df.drop('Type', axis=1, inplace=True)

# Class 47 Dun 14/02/24 ... Class 50 Gal ..
df.loc[(df.Class==47)|(df.Class==50), 'Class'] = np.nan

df['AgeMin'] = [ float(x[:x.find('Y')]) for x in df.AgeLimit ]
df['AgeMax'] = [ float(x[:x.find('Y')]) if x.endswith('only') 
                else float(x[x.rfind('Y')-1]) if x.find('to') > 0    # < 10
                else np.nan for x in df.AgeLimit ]

# df[['RaceId','Class','AgeLimit','AgeMin','AgeMax']]
df.drop('AgeLimit', axis=1, inplace=True)




Going_ = df['Going'].unique()
Going_ = Going_[Going_.argsort()]
print(Going_)

# AW False -> Turf for GB/IRE
df['AW'] = df.Going.str.contains('AW')

Going__ = [ fGoing(x) for x in Going_ ]

df['GoingNum'] = df['Going'].apply(fGoing).apply(fGoingNum)


# 4          Race  string[python]         0
# 5          Type  string[python]    661257
# 6         Class         float64    274019
# 7      AgeLimit  string[python]         0
# 8         Prize         float64         0
# 9           Ran            int8         0
# 10        Yards           int16         0
# 11        Going  string[python]        42
# 12        Limit         float64    481391
# 13  RaceWinTime         float64       226
# 14         FPos  string[python]         6
# 15       DstBtn  string[python]    180227
# 16     TotalBtn  string[python]     63200
# 17       CardNo         float64      5581
# 18    HorseName  string[python]         0
# 19         Draw         float64    430520
# 20           Sp         float64       293
# 21          Age            int8         0
# 22        Stone            int8         0
# 23          Lbs            int8         0
# 24    WeightLBS            int8         0
# 25         Favs          object    848284
# 26          Aid  string[python]    686765
# 27      Trainer  string[python]         2
# 28       Jockey  string[python]         1
# 29        Allow         float64    792251
# 30           OR  string[python]    285687
# 31     Comments  string[python]      2970
