import pickle

with open('/Users/islamtayeb/Documents/spotify-clustering/analysis/outputs/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['combined']['dataframe']

print("Column names in dataframe:")
for col in df.columns:
    print(f"  • {col}")