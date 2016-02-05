import os
import pandas
import mlutilities.types as mltypes
import mlutilities.dataTransformation as mldata
import mlutilities.modeling as mlmodel
import mlutilities.utilities as mlutils
import thesisFunctions


hucsFile = '../SacramentoData/Sacramento_basin_huc12_v2.csv'
hucRegionsFile = '../SacramentoData/Sacramento_huc12_ecoregions.csv'

hucs = pandas.read_csv(hucsFile)
hucRs = pandas.read_csv(hucRegionsFile)

hucRs = hucRs.loc[:, ['HUC_12', 'AggEcoreg']]

hucRs.rename(columns={'HUC_12':'HUC12'}, inplace=True)

common = hucs.merge(hucRs, on=['HUC12'])
common.drop_duplicates(subset='HUC12', inplace=True)

