import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

print("welcome to the hotel recommendation system!" )
print("made by Manas Saini (25BCE10183)")
df = pd.read_csv("Hotel Landmarks.csv")
df.columns = df.columns.str.strip()
df['Type'] = df['Type'].fillna('Unknown').str.strip().str.lower()

print("Building hotel profiles...")

FEATURE_TYPES = {
    'beach':      ['beach'],
    'religious':  ['religious', 'religious place', 'hindu temple', 'mosque', 'church', 'place of worship'],
    'recreation': ['recreation'],
    'shopping':   ['shopping', 'shopping mall', 'mall'],
    'museum':     ['museum'],
    'restaurant': ['restaurant'],
    'amusement':  ['amusement park'],
    'zoo':        ['zoo'],
    'monument':   ['monument'],
    'art_gallery':['art gallery'],
    'hospital':   ['hospital'],
    'university': ['university'],
}

records = []
for code, group in df.groupby('Code'):
    types = group['Type'].tolist()
    nearby = group[group['Distance'] <= 5.0]
    avg_dist = nearby['Distance'].mean() if len(nearby) > 0 else 999.0

    row = {
        'hotel_code': str(code),
        'city':       group['City'].iloc[0],
        'lat':        group['Latitude'].mean(),
        'lon':        group['Longitud'].mean(),
        'landmark_count': len(group),
        'avg_nearby_dist': round(avg_dist, 2),
    }
    for feat, keywords in FEATURE_TYPES.items():
        row[f'has_{feat}'] = int(any(t in keywords for t in types))

    records.append(row)

hotels = pd.DataFrame(records)
print(f"  → {len(hotels)} hotels across {hotels['city'].nunique()} cities\n")

FEAT_COLS = [f'has_{f}' for f in FEATURE_TYPES]

X = hotels[FEAT_COLS].values.astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def recommend(city=None, preferences=None, max_dist=5.0, top_n=10):
    """
    Parameters
    ----------
    city        : str or None  — filter by city name (e.g. 'Manali')
    preferences : list or None — list of features you want nearby
                  options: beach, religious, recreation, shopping, museum,
                           restaurant, amusement, zoo, monument, art_gallery,
                           hospital, university
    max_dist    : float        — max avg distance to landmarks in km
    top_n       : int          — number of results to return
    """
    pool = hotels.copy()

    # Filter by city
    if city:
        pool = pool[pool['city'].str.lower() == city.lower()]
        if pool.empty:
            print(f"No hotels found for city: {city}")
            return pd.DataFrame()

    filtered = pool[pool['avg_nearby_dist'] <= max_dist]
    if filtered.empty:
        print(f"No hotels within {max_dist}km — showing closest available.")
        filtered = pool


    if preferences:
        user_vec = np.zeros((1, len(FEAT_COLS)))
        for p in preferences:
            key = f'has_{p}'
            if key in FEAT_COLS:
                user_vec[0, FEAT_COLS.index(key)] = 1

        hotel_vecs = filtered[FEAT_COLS].values.astype(float)
        scores = cosine_similarity(user_vec, hotel_vecs)[0]
    else:
        scores = np.ones(len(filtered))

    lc_norm   = np.log1p(filtered['landmark_count'].values) / np.log1p(50)
    dist_norm = np.clip(1 - filtered['avg_nearby_dist'].values / max_dist, 0, 1)
    final_score = scores * 0.6 + lc_norm * 0.25 + dist_norm * 0.15

    filtered = filtered.copy()
    filtered['match_score'] = np.round(final_score * 100, 1)
    filtered['cos_sim']     = np.round(scores, 3)

    result = filtered.sort_values('match_score', ascending=False).head(top_n)
    cols = ['hotel_code', 'city', 'landmark_count', 'avg_nearby_dist', 'match_score', 'cos_sim'] + FEAT_COLS
    return result[cols].reset_index(drop=True)


if __name__ == "__main__":

    print("=" * 60)
    print("Example 1: Beach + Restaurant hotels in Goa-style cities")
    print("=" * 60)
    result1 = recommend(
        city=None,
        preferences=['beach', 'restaurant'],
        max_dist=5.0,
        top_n=10
    )
    print(result1[['hotel_code', 'city', 'landmark_count', 'avg_nearby_dist', 'match_score']].to_string())

    print("\n" + "=" * 60)
    print("Example 2: Religious + Museum hotels in Varanasi")
    print("=" * 60)
    result2 = recommend(
        city='Varanasi',
        preferences=['religious', 'museum', 'monument'],
        max_dist=10.0,
        top_n=10
    )
    print(result2[['hotel_code', 'city', 'landmark_count', 'avg_nearby_dist', 'match_score']].to_string())

    print("\n" + "=" * 60)
    print("Example 3: Shopping + Amusement in Mumbai")
    print("=" * 60)
    result3 = recommend(
        city='Mumbai',
        preferences=['shopping', 'amusement', 'restaurant'],
        max_dist=8.0,
        top_n=10
    )
    print(result3[['hotel_code', 'city', 'landmark_count', 'avg_nearby_dist', 'match_score']].to_string())

    print("\n" + "=" * 60)
    print("Available cities:")
    print(sorted(hotels['city'].unique()))
    print("\nAvailable preferences:", list(FEATURE_TYPES.keys()))