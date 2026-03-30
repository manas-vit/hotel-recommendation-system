# Hotel Recommendation System

A content-based hotel recomendation engine built in Python. Given a destination city and your preferences (beach, museum, shopping, etc.), the system ranks hotels by how well their surrounding landmarks match what you are looking for.

---

## What It Does

Most hotel search tools rank by price or ratings. This system ranks by **surroundings** — it looks at the types of landmarks near each hotel and matches them against your stated preferences using cosine similarity.

- 11,484 hotels across 70 cities in India
- 12 preference categories (beach, religious, shopping, museum, and more)
- Configurable distance filter — only consider landmarks within X km
- No user accounts or ratings neded — works entirely on landmark data

---

## How It Works

The project uses **content-based filtering** with **cosine simillarity**:

1. **Feature Engineering** — Each hotel's raw landmark records are aggregated into a 12-dimensional binary feature vector (e.g. `has_beach=1`, `has_museum=0`)
2. **User Vector** — Your selected preferences are turned into a matching vector of the same shape
3. **Cosine Similarity** — The angle between the user vector and each hotel's vector gives a match score
4. **Final Ranking** — Match score (60%) + landmark richnes bonus (25%) + proximity bonus (15%)

---

## Project Structure

```
hotel-recommendation/
├── hotel_recommender.py     # Main script — run this
├── Hotel Landmarks.csv      # Dataset (place in same folder)
└── README.md
```

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

---

## Setup

1. Clone this repository:

```bash
git clone https://github.com/[YOUR-USERNAME]/hotel-recommendation.git
cd hotel-recommendation
```

2. Place `Hotel Landmarks.csv` in the same directory as `hotel_recommender.py`

3. Install packages:

```bash
pip install pandas numpy scikit-learn
```

---

## Usage

Run the script directly to see three example outputs:

```bash
python hotel_recommender.py
```

Or import `recommend()` in your own code:

```python
from hotel_recommender import recommend

# Beach + dining hotels anywhere in India
results = recommend(
    city=None,
    preferences=['beach', 'restaurant'],
    max_dist=5.0,
    top_n=10
)

# Religious + cultural hotels in Varanasi
results = recommend(
    city='Varanasi',
    preferences=['religious', 'museum', 'monument'],
    max_dist=10.0,
    top_n=10
)

print(results[['hotel_code', 'city', 'landmark_count', 'avg_nearby_dist', 'match_score']])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `city` | str or None | None | Filter by city. `None` searches all 70 cities |
| `preferences` | list | None | Landmark types you want nearby |
| `max_dist` | float | 5.0 | Max average distance to landmarks in km |
| `top_n` | int | 10 | Number of results to return |

### Available Preferences

```
beach, religious, recreation, shopping, museum,
restaurant, amusement, zoo, monument, art_gallery,
hospital, university
```

### Available Cities (sample)

Agra, Alleppey, Chandigarh, Coorg, Darjeeling, Jaisalmer, Jodhpur, Manali,
Mumbai, Munnar, Mussoorie, Mysore, Nainital, Ooty, Pondicherry, Rishikesh,
Shimla, Srinagar, Varanasi, Visakhapatnam, Wayanad, udaipur, and 48 more.

Run the script once to print all 70 city names.

---

## Sample Output

```
============================================================
Religious + Museum + Monument hotels in Varanasi
============================================================
         hotel_code      city  landmark_count  avg_nearby_dist  match_score
0  8559213960724...  Varanasi              25             2.69         83.6
1  5143267348040...  Varanasi              27             1.48         80.4
2  1727493204134...  Varanasi              27             1.64         80.2
```

---

## Dataset

**Hotel Landmarks CSV** — 264,054 records of hotel-to-landmark relationships.

| Column | Description |
|--------|-------------|
| Code | Unique hotel identifier |
| City | City name |
| Landmarks | Name of a nearby landmark |
| Distance | Distance from hotel to landmark (km) |
| Type | Landmark category (Beach, Museum, Hospital, etc.) |
| Latitude / Longitud | Geographic cordinates |

---

## Author

**MANAS SAINI**  
CSE(CORE)
VIT BHOPAL UNIVERSITY 

---

## License

Built as part of an AI/ML course assignment. Dataset provided as course material.
