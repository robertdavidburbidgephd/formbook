# AW Surface lookup 2024
# Sth Fibresand before 7 Dec 2021
# Wol Fibresand before 2 Oct 2004 ; Polytrack before 11 Aug 2018
# NB. No spaces in CourseName here TOFIX
z = [ y.partition('|') for y in set(
    d24[d24.AW][['CourseName','Going']].apply(lambda r: r[0] + '|' +\
            r[1].replace('Standard','').replace('/ Slow','').replace('AW','').replace('(','')\
                .replace(')','').replace('-','').replace(' ',''),
                axis=1).unique()) ]
# d24.drop('Going', axis=1, inplace=True)

AWSurfaceTable2024 = pd.DataFrame.from_dict({'CourseName': [x[0] for x in z], 'Surface': [x[2] for x in z]})
AWSurfaceTable2024 = AWSurfaceTable2024[AWSurfaceTable2024.Surface != '']   # CourseName = 'Southell (AW)'

AWHandedness = [ ('Chelmsford City', 'LH'), ('Kempton', 'RH'), ('Newcastle', 'LH'), ('Dundalk','LH')
              , ('Southwell','LH'), ('Lingfield','LH'), ('Laytown',''), ('Wolverhampton','LH') ]
AWHandednessTable = pd.DataFrame.from_dict({'CourseName': [x[0] for x in AWHandedness],
                                          'Handedness': [x[1] for x in AWHandedness]})

AWTable = AWSurfaceTable2024.merge(AWHandednessTable)







CoolaghMagic = d24[d24.HorseName=='Coolagh Magic'][['RaceDate','CourseName','Yards','GoingNum','OR']].\
                merge(AWTable)
CoolaghMagic = CoolaghMagic.iloc[CoolaghMagic.RaceDate.argsort()]

# CoolaghMagic['dOR'] = CoolaghMagic.OR.diff(1)

# import statsmodels.api as sm

# data_endog = CoolaghMagic['dOR'].values[1:]
# data_exog = np.concatenate([CoolaghMagic[['Yards','GoingNum']].values[1:,], np.ones([len(data_endog),1])], axis=1)

# rlm_model = sm.RLM(data_endog, data_exog, M=sm.robust.norms.HuberT())
# rlm_results = rlm_model.fit()

# print(rlm_results.params)