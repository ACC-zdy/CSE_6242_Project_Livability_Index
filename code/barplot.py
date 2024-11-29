import pandas as pd
df = pd.read_csv('data_.csv')
result_df=pd.read_csv('data_.csv')

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


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

filtered_df = df[df['Country'].isin(filtered_country_list)][['Country'] + column_titles]

long_df = pd.melt(filtered_df, id_vars=['Country'], value_vars=column_titles,
                  var_name='Indicator', value_name='Value')

plt.figure(figsize=(10, 6))
sns.barplot(x='Indicator', y='Value', hue='Country', data=long_df,palette='Blues')

plt.title('Comparison of Countries for Each Indicator')
plt.xlabel('Indicator')
plt.ylabel('Value')

plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig('barplot.png', dpi=300, bbox_inches='tight')

plt.show()
