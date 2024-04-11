### Time-Spatial data preprocessing
"""
This section is aims to process the data which contains both spatial and time information.
- road_case
- sinkhole_case
- pipe_case
"""

# Import packages
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
import time
import pytz
import shapely
from shapely.geometry import Point, Polygon
from shapely import LineString
from shapely.wkt import loads
from datetime import datetime, timedelta
from pyproj import Proj
import pyproj
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import xml.etree.ElementTree as ET
import dask_geopandas as dgpd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from function import *
import sys

# 從命令行參數中讀取值
time_window = int(sys.argv[1])
time_pred = sys.argv[2]
time_pred = pd.to_datetime(time_pred).date()
time_start = time_pred - timedelta(days=time_window+1)
time_end = time_pred - timedelta(days=1)

### Config
ROOT_PATH = r"D:\Chu's Document!\02 Project\06 道路塌陷防治專案(天坑)"

path_gpkg = ROOT_PATH + r"\03 Data\Model_building\training_data_TP_temp.gpkg"
hex_5m_rd_sample_cn = dgpd.read_file(path_gpkg, chunksize=1000000)
hex_5m_rd_sample_cn = hex_5m_rd_sample_cn.compute()

# 108-112年新工處資料(璿達去重&威竹加標記)
road_case_raw = pd.read_csv(ROOT_PATH + r"\03 Data\Processed\道管系統坑洞案件_108-112_Chu加案件標記_20240201.csv")
road_case = road_case_raw.copy()

# 保留所需欄位並去除缺少經緯度資料
road_keep_col = [
    '案件編號', '查報日期', '道路寬度', 'link', 'lng_twd97', 
    'lat_twd97', '疑似天坑案件', 'TUIC天坑判斷'
]
road_case = road_case[road_keep_col]
road_case = road_case.dropna(subset=["lng_twd97"])

# 道管案件數量計算
road_case_out = road_case[road_case["TUIC天坑判斷"] != 2]
road_case_in = road_case[road_case["TUIC天坑判斷"] == 2]

road_case_out_gdf = create_buffered_gdf(road_case_out, 'lng_twd97', 'lat_twd97', buffer_distance=5)
road_case_in_gdf = create_buffered_gdf(road_case_in, 'lng_twd97', 'lat_twd97', buffer_distance=5)

# transfer to date
road_case_out_gdf["查報日期"] = pd.to_datetime(road_case_out_gdf["查報日期"])
road_case_out_gdf["查報日期"] = road_case_out_gdf["查報日期"].dt.date
road_case_in_gdf["查報日期"] = pd.to_datetime(road_case_in_gdf["查報日期"])
road_case_in_gdf["查報日期"] = road_case_in_gdf["查報日期"].dt.date

# calculate case during time period
hex_5m_rd_sample_cn = calculate_case_during_period(
    hex_5m_rd_sample_cn, road_case_out_gdf, "查報日期", time_pred, time_window=time_window,
    count_column="road_case_count"
)
hex_5m_rd_sample_cn = calculate_case_during_period_boolean(
    hex_5m_rd_sample_cn, road_case_in_gdf, "查報日期", time_pred, time_window=time_window,
    count_column="sinkhole_case_count"
) # 有案件為1，無案件為0

### Pipe_case
# 水系案件清單：108-112
pipe_case_raw = pd.read_excel(ROOT_PATH + r"\03 Data\Raw\水系搶修結案_施工日期110-1120831_1120912(作業用)＿OK.xlsx")
pipe_case = pipe_case_raw.copy()

# 保留所需欄位並去除缺少經緯度資料
pipe_keep_col = ['許可證編號', '預定挖掘起日', '預定挖掘迄日', 'X座標值', 'Y座標值']
pipe_case = pipe_case[pipe_keep_col]
pipe_case = pipe_case.dropna(subset=["X座標值"])

# transfer to date
pipe_case["預定挖掘起日"] = rocdate_transfer_to_time(pipe_case["預定挖掘起日"])
pipe_case["預定挖掘迄日"] = rocdate_transfer_to_time(pipe_case["預定挖掘迄日"])

pipe_case_gdf = create_buffered_gdf(pipe_case, 'X座標值', 'Y座標值', buffer_distance=5)
pipe_case_gdf["預定挖掘起日"] = pd.to_datetime(pipe_case["預定挖掘起日"]).dt.date

hex_5m_rd_sample_cn = calculate_case_during_period(
    hex_5m_rd_sample_cn, pipe_case_gdf, "預定挖掘起日", time_pred, count_column="pipe_case_count"
)

#### Add Timestamp
hex_5m_rd_sample_cn["Baseline_Date"] = time_pred

### Time data preprocessing
"""
This section is aims to process the data which contains only time information.
- precipitation
"""
time_data_path = ROOT_PATH + r"\03 Data\Model_building\time_series_data.csv"
time_data = pd.read_csv(time_data_path)
time_data["date"] = pd.to_datetime(time_data["date"])
time_data["date"] = time_data["date"].dt.date

# calculate precipitation
hex_5m_rd_sample_cn = process_rainfall_data(
    time_data, time_start, time_pred, hex_5m_rd_sample_cn, column='precipitation'
)

# Calculate earthquakes count
hex_5m_rd_sample_cn = process_sum_data(
    time_data, time_start, time_pred, hex_5m_rd_sample_cn, column='earthquakes_count'
)

# Calculate MeanTideLevel
hex_5m_rd_sample_cn = process_mean_data(
    time_data, time_start, time_pred, hex_5m_rd_sample_cn, column='meantidelevel'
)

# Calculate MeanTideLevel
hex_5m_rd_sample_cn = process_mean_data(
    time_data, time_start, time_pred, hex_5m_rd_sample_cn, column='meanhighwaterlevel'
)

# Calculate MeanTideLevel
hex_5m_rd_sample_cn = process_mean_data(
    time_data, time_start, time_pred, hex_5m_rd_sample_cn, column='meanlowwaterlevel'
)

# create year and month columns
hex_5m_rd_sample_cn["year"] = pd.to_datetime(hex_5m_rd_sample_cn["Baseline_Date"]).dt.year
hex_5m_rd_sample_cn["month"] = pd.to_datetime(hex_5m_rd_sample_cn["Baseline_Date"]).dt.month

# create lng lat columns
hex_5m_rd_sample_cn["lng"] = hex_5m_rd_sample_cn.centroid.apply(extract_lng)
hex_5m_rd_sample_cn["lat"] = hex_5m_rd_sample_cn.centroid.apply(extract_lat)

# reorder columns
new_order = [
    'id', 'centroid', 'TNAME', 'soil_liquid_class', 'width', 'road_name',
    'road_id', 'sp_count', 'rp_count', 'rd_count', 'cn_count', 'wp_count',
    'pipe_count', 'geometry', 'road_case_count', 'pipe_case_count', 'park_area', 
    'school_greening_area', 'riverside_highland_area', 'birds_conservation_area',
    'agriculture_area', 'pavement_sidewalk_area', 'pavement_parkinglot_area', 
    'pavement_park_area', 'pavement_school_area', 'pavement_pac_area', 'building_floor',
    'building_area', 'building_volume', 'underground_floor', 'underground_mrt', 'underground_hsr', 
    'underground_tr', 'Baseline_Date', 'precipitation', 'earthquakes_count', 'meantidelevel', 
    'meanhighwaterlevel', 'meanlowwaterlevel', 'year', 'month', 'lng', 'lat', 'sinkhole_case_count'
]
hex_5m_rd_sample_cn = hex_5m_rd_sample_cn[new_order]

### Output

# Path setting
folder_root_path = ROOT_PATH + r"\03 Data\Model_building\TP_2022"

# transfer to string
time_pred = str(time_pred)
time_window = str(time_window)
hex_5m_rd_sample_cn["Baseline_Date"] = hex_5m_rd_sample_cn["Baseline_Date"].astype(str)

# Split by town
# town_list = hex_5m_rd_sample_cn['TNAME'].unique().tolist()
# town_dict = {
#     '北投區': 'BT', '士林區': 'SL', '大同區': 'DT', '萬華區': 'WH', '中正區': 'ZZ', '中山區': 'ZS',
#     '大安區': 'DA', '松山區': 'SS', '內湖區': 'NH', '文山區': 'WS', '信義區': 'XY', '南港區': 'NG'
# }

# for i in town_list:
#     data_export = hex_5m_rd_sample_cn[hex_5m_rd_sample_cn['TNAME'] == i]

#     # Export to csv by town
#     csv_path = folder_root_path + f"/Hex_TP_{town_dict[i]}_{time_window}_{time_pred}.csv"    
#     data_export.to_csv(csv_path)

    # Export to GeoPackage by town
    # gpkg_path = folder_root_path + f"/Hex_TP_{town_dict[i]}_{time_window}_{time_pred}.gpkg"
    # data_export.to_file(gpkg_path, driver="GPKG")

# Export to csv in city scale
csv_path = folder_root_path + f"/Hex_TP_{time_window}_{time_pred}.csv"    
hex_5m_rd_sample_cn.to_csv(csv_path)

if __name__ == "__main__":
    # 呼叫函式並顯示結果
    print("Export Files Successfully!")
