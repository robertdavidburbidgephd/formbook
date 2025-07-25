import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import argparse

## command-line args
# python src/xmast_example.py <CourseName> ... ???

## named args or config ... <- multiple dist/going , defaults
parser = argparse.ArgumentParser(
                    prog='xmast',
                    description='Create drawbias xmast chart',
                    epilog='')
parser.add_argument('-c', '--coursename')
parser.add_argument('-d', '--distance', nargs='+')
parser.add_argument('-g', '--going', nargs='+')
args = parser.parse_args()

course_name = args.coursename
dist = [ float(x)*220 for x in args.distance ]
going_dict = {'f': 'firm', 'gf': 'goodtofirm', 'stf': 'standard/fast', 'gd': 'good',
             'st': 'standard', 'gdy': 'goodtoyielding', 'y': 'yielding',
             'gs': 'goodtosoft', 'stsl': 'standard/slow', 'ysft': 'yieldingtosoft',
             'sft': 'soft', 'slw': 'slow', 'sfthy': 'softtoheavy', 'hy': 'heavy'}
going = [ going_dict[x] for x in args.going ]

# print(course_name)
# print(dist)
# print(going)



## Get the formbook data
fpath = 'data/prep/'
dfile = 'form_data_v2.0.csv'

form_df = pd.read_csv( fpath + dfile, header=0, parse_dates=True )

## Prep the formbook data
from formbook_defs import prexmast
prexmast_df = prexmast(form_df)

## Call xmast
min_ave_supp = 2
# course_name = 'Sandown'

# dist = [ 7*220 , 8*220 ]
# going = ['good', 'goodtosoft']

from formbook_defs import xmast
hm_data, hm_supp, hm_annot_index, hm_annot_supp = xmast(prexmast_df, course_name, dist=dist, going=going, 
                                                        stalls='far side', min_ave_supp=min_ave_supp)

""" def plotxmast """

plt.figure()
plt.title('Draw Bias (Return), ' + course_name + ' (' + str(dist) + 'yards) ' + str(going))

sns.heatmap(hm_data.dropna(axis=0, how='all'),
    cmap='gist_earth_r', robust=True,
    # annot=hm_annot_index.dropna(axis=0, how='all'), 
    # annot=hm_annot_supp.dropna(axis=0, how='all'),
            fmt='', annot_kws={'fontsize': 'xx-small'},
    annot=(hm_annot_index + '\n(' + hm_annot_supp + ')').dropna(axis=0, how='all'),
            xticklabels=False, yticklabels=2
    )
# plt.show()
plt.savefig('xmast_' + course_name + '.png')
plt.close()

# """ loop call + plot xmast """

# goings = [ [ 'goodtofirm', 'good' ] , [ 'goodtosoft', 'soft', 'heavy' ] ]

# # group dist by starting stalls posn
# dists = [ [ 5.5*220 ] , [ 13*220+89, 5*220+16 ] , [ 12*220+66 ] , [ 11*220+79 ] , 
#          [ 10*220+75, 18*220+147 ] , [ 15*220+195, 7*220+122 ] , [ 7*220+2] , [ 6*220+18] ]

# min_ave_supp = 1
# course_name = 'Chester'

# for dist in dists:
#     for going in goings:

#         plt.figure()
#         plt.title('Draw Bias (Return), ' + course_name + ' (' + str(dist) + 'yards) ' + str(going))
#         hm_data, hm_supp, hm_annot_index, hm_annot_supp = xmast(df, course_name, dist=dist, going=going, 
#                                                                 stalls='inside', min_ave_supp=min_ave_supp)
#         sns.heatmap(hm_data.dropna(axis=0, how='all'),
#             cmap='gist_earth_r', robust=True,
#             # annot=hm_annot_index.dropna(axis=0, how='all'), 
#             # annot=hm_annot_supp.dropna(axis=0, how='all'),
#                     fmt='', annot_kws={'fontsize': 'xx-small'},
#             annot=(hm_annot_index + '\n(' + hm_annot_supp + ')').dropna(axis=0, how='all'),
#                     xticklabels=False, yticklabels=2
#             )
#         # plt.show()
#         plt.savefig('xmast_' + course_name + '_' + str(dist) + '_' + str(going) + '.png')
#         plt.close()