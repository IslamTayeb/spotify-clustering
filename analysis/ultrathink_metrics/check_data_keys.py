import pickle

with open('/Users/islamtayeb/Documents/spotify-clustering/analysis/outputs/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("Available keys in analysis_data.pkl:")
for key in data.keys():
    print(f"  • {key}")
    if hasattr(data[key], 'shape'):
        print(f"    Shape: {data[key].shape}")
    elif hasattr(data[key], '__len__'):
        print(f"    Length: {len(data[key])}")