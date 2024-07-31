from .linear_projection import LinearProjectionNDim
from .data_samplers import RandomTpRangeSubsetDataset, RandomTimeDelaySubsetDataset, RandomXYSubsetDataset

from .imd_nd import IMD_nD
from .imd_1d import IMD_1D
from .imd_1d_smap import IMD_1D_smap
from .imd_nd_smap import IMD_nD_smap
from .imd_reg import IMD_reg

class IMD:
     IMD_nD = IMD_nD
     IMD_1D = IMD_1D
     IMD_1D_smap = IMD_1D_smap
     IMD_nD_smap = IMD_nD_smap
     IMD_reg = IMD_reg