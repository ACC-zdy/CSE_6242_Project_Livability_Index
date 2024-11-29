import pandas as pd
df = pd.read_csv('data_.csv')
result_df=pd.read_csv('allData.csv')

unique_years = result_df['Year'].unique()
unique_years=unique_years[::-1]

yearly_data = result_df.groupby('Year')

year_to_dataset_dict = {year: group for year, group in yearly_data}

N_country = 40

year = unique_years[0]
dataset = year_to_dataset_dict[year]

top_n_countries_df = dataset.nlargest(N_country, 'Quality of Life Index')

top_n_countries_list = top_n_countries_df['Country'].tolist()
print(top_n_countries_list)

filtered_country_list = []
for country in top_n_countries_list:
    is_in_all_datasets = True

    for year in unique_years:
        ds = year_to_dataset_dict[year]

        if country not in ds['Country'].values:
            is_in_all_datasets = False
            break

    if is_in_all_datasets:
        filtered_country_list.append(country)

print(filtered_country_list)

begin_N = N_country
while (1):
    begin_N += 1
    country = dataset.nlargest(begin_N, 'Quality of Life Index')['Country'].iloc[-1]
    is_in_all_datasets = True

    for year in unique_years:
        ds = year_to_dataset_dict[year]

        if country not in ds['Country'].values:
            is_in_all_datasets = False
            break

    if is_in_all_datasets:
        filtered_country_list.append(country)
    if len(filtered_country_list) >= N_country:
        break

print(filtered_country_list)
filtered_country_list = ['Denmark', 'Switzerland', 'Finland', 'Japan', 'Qatar', 'United Arab Emirates', 'United States',
                         'Canada', 'Australia', 'New Zealand']



column_titles = dataset.columns[2:7].tolist()

import matplotlib.pyplot as plt
country_data = {country: [] for country in filtered_country_list}
for year in unique_years:
    year_data = year_to_dataset_dict[year]
    for country, quality_of_life in zip(year_data['Country'], year_data['Quality of Life Index']):
        if country in filtered_country_list:
            country_data[country].append(quality_of_life)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plot_data = []

unique_years_list = unique_years.tolist()

for year in unique_years_list:
    for country, quality_of_life_values in country_data.items():
        print(country,quality_of_life_values)
        idx = np.where(unique_years == year)[0][0]
        plot_data.append({'Year': year, 'Country': country, 'Quality of Life Index': quality_of_life_values[idx]})

plot_df = pd.DataFrame(plot_data)

plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df, x='Year', y='Quality of Life Index', hue='Country', palette="Blues", markers=True)

plt.xlabel('Year')
plt.ylabel('Quality of Life Index')
plt.title('Quality of Life Index by Country')
plt.legend(title='Country')
plt.xticks(rotation=45)
plt.savefig('lineplot.png', dpi=300, bbox_inches='tight')

plt.show()




