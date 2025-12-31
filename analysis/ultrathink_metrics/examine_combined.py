import pickle

with open('/Users/islamtayeb/Documents/spotify-clustering/analysis/outputs/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys in 'combined' section:")
for key in data['combined'].keys():
    print(f"  • {key}")
    if hasattr(data['combined'][key], 'shape'):
        print(f"    Shape: {data['combined'][key].shape}")
    elif hasattr(data['combined'][key], '__len__') and not isinstance(data['combined'][key], (str, dict)):
        print(f"    Length: {len(data['combined'][key])}")