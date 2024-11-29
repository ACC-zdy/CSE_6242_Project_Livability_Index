import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# 使用方法 1，路径前加 r
shapefile_path = r"C:\Users\m1527\Desktop\ne_110m_admin_0_countries\ne_10m_admin_0_countries_chn\ne_10m_admin_0_countries_chn.shp"

# 加载世界地图数据
world = gpd.read_file(shapefile_path)

# 加载宜居指数数据
df = pd.read_csv(r"C:\Users\m1527\Desktop\123.csv")

# 确保地图和 CSV 中的国家名称一致：标准化大小写
world['NAME'] = world['NAME'].str.strip().str.title()  # 删除多余空格，标题化格式
df['Country'] = df['Country'].str.strip().str.title()  # 删除多余空格，标题化格式
# 合并数据
merged = world.set_index('NAME').join(df.set_index('Country'))

'''
# 统计是否能成功匹配
matched_countries = 0
total_world_countries = 0
total_csv_countries = 0

# 检查和统计
for country in world['NAME']:
    total_world_countries += 1
    if country in df['Country'].values:
        matched_countries += 1

for country in df['Country']:
    total_csv_countries += 1
    if country in world['NAME'].values:
        matched_countries += 1

# 打印匹配结果
print(f"Total countries in world shapefile: {total_world_countries}")
print(f"Total countries in CSV data: {total_csv_countries}")
print(f"Matched countries: {matched_countries}")

# 进一步检查每个国家名称
print("\nCountries in world shapefile (unique):")
print(world['NAME'].unique())

print("\nCountries in CSV data (unique):")
print(df['Country'].unique())
'''
# 绘图
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# 使用红-蓝渐变（RdBu_r）并设置国界线为灰色细线
# NaN 值显示为白色，通过设置 'NaN' 为 'white' 来实现
cmap = 'summer_r'
merged.plot(column='Quality_of_Life_Index', ax=ax, legend=False,
            cmap=cmap,  # 红-蓝渐变（低值红色，高值蓝色）
            edgecolor='gray',  # 设置国界线颜色为灰色
            linewidth=0.2,  # 线条宽度
            missing_kwds={'color': 'white'})  # 设置无数据国家为白色

# 为所有国家添加边界（即使是没有数据的国家也显示边界）
world.boundary.plot(ax=ax, color='gray', linewidth=0.7)  # 添加所有国家的边界，灰色

# 去掉外框线和坐标轴
ax.set_axis_off()  # 去掉坐标轴和外框线

# 去掉标题（如果你不需要标题的话）
ax.set_title('')  # 可以将标题设置为空，或者根本不调用

# 添加颜色条（图例），并将其位置设置到左下角
# 使用 colorbar() 手动调整位置
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=merged['Quality_of_Life_Index'].min(), vmax=merged['Quality_of_Life_Index'].max()))
sm.set_array([])  # 空数组用于生成色条
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

# 设置图例位置到左下角
cbar.ax.set_position([0.16, 0.4, 0.02, 0.2])  # 调整位置，0.0 是 x 位置，-0.1 是 y 位置，0.2 是宽度，0.02 是高度

# 保存图像为 PNG
output_path = r'C:\Users\m1527\Desktop\GT\map.png'  # 输出路径
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 设置高分辨率，去掉空白边距
# 显示图表
plt.show()
