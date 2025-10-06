import sys
from unittest.mock import patch
from urllib3.util.retry import Retry


# ========= PHASE A : COLLECTE BRUTE (SANS FILTRAGE) =========
# Patch la classe Retry pour accepter method_whitelist
original_init = Retry.__init__

def patched_init(self, *args, **kwargs):
    if 'method_whitelist' in kwargs:
        kwargs['allowed_methods'] = kwargs.pop('method_whitelist')
    return original_init(self, *args, **kwargs)

Retry.__init__ = patched_init

# Maintenant on peut importer pytrends
from pytrends.request import TrendReq
import pandas as pd
import time
import re
from tqdm import tqdm

# ---- 0) TES SEEDS ----
seed_terms = [
    "moroccan food", "restaurant in morocco",
    "morocco hotels", "marrakech hotels", "agadir hotels", "tangier hotels", 
    "chefchaouen hotels", "riad marrakech", "riad fes",
    "flights to morocco", "royal air maroc", "marrakech flights", 
    "casablanca flights", "agadir flights", "tangier flights", "fes flights", "morocco map",
    "morocco travel", "visit morocco", "morocco tourism", "things to do in morocco",
    "sahara desert tour", "merzouga tour", "atlas mountains tour", 
    "essaouira day trip", "ouzoud waterfalls", "ait benhaddou", "places in morocco",
    "shopping in morocco", "marrakech souk", "hammam morocco", "argan oil morocco",
    "morocco weather", "marrakech weather", "agadir weather", 
    "climate of morocco", "best time to visit morocco",
    "morocco visa", "morocco itinerary", "morocco safety"
]

# ---- 1) Paramètres optimisés ----
GEOS = ["", "US", "GB", "FR", "ES", "DE"]
CATS = [0, 67]
TIMEFRAMES = ["today 12-m", "today 5-y", "2020-01-01 2024-12-31"]
TOP_K = 25
SLEEP = 1.5
MAX_DEPTH = 2

pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)

def related_union(term):
    """Union des Related (Top+Rising) sur (geo, cat, timeframe)."""
    out = set()
    success_count = 0
    
    for tf in TIMEFRAMES:
        for geo in GEOS:
            for cat in CATS:
                try:
                    pytrends.build_payload([term], geo=geo, cat=cat, timeframe=tf, gprop="")
                    time.sleep(SLEEP)
                    
                    rq = pytrends.related_queries()
                    d = rq.get(term, {})
                    
                    for kind in ("top", "rising"):
                        df = d.get(kind)
                        if df is not None and len(df) > 0:
                            success_count += 1
                            qs = df["query"].astype(str).head(TOP_K).tolist()
                            cleaned = [re.sub(r"\s+", " ", q.lower().strip()) for q in qs if q.strip()]
                            out.update(cleaned)
                    
                except Exception as e:
                    if "429" not in str(e) and "quota" not in str(e).lower():
                        print(f"\n  ⚠ [{term[:30]}]: {str(e)[:80]}")
                    time.sleep(SLEEP * 2)
                    continue
    
    if success_count > 0:
        print(f"  ✓ {term[:40]:40} → {len(out)} related")
    else:
        print(f"  ✗ {term[:40]:40} → 0 related")
    
    return list(out)

# ---- 2) Depth 1
print("\n" + "="*70)
print("DEPTH 1: Collecting related queries from seed terms")
print("="*70 + "\n")

frontier = list(dict.fromkeys([t.strip() for t in seed_terms]))
all_terms = set(frontier)
depth1_new = []

for term in tqdm(frontier, desc="Depth 1"):
    rel = related_union(term)
    for r in rel:
        if r and r not in all_terms:
            all_terms.add(r)
            depth1_new.append(r)

# ---- 3) Depth 2
print("\n" + "="*70)
print(f"DEPTH 2: Collecting from {len(depth1_new)} new terms")
print("="*70 + "\n")

frontier = depth1_new[:]
depth2_new = []

if len(frontier) > 0:
    for term in tqdm(frontier, desc="Depth 2"):
        rel = related_union(term)
        for r in rel:
            if r and r not in all_terms:
                all_terms.add(r)
                depth2_new.append(r)
else:
    print("⚠️  No new terms from Depth 1, skipping Depth 2")

# ---- 4) Résultats
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Seeds: {len(seed_terms)}")
print(f"Depth1 new terms: {len(depth1_new)}")
print(f"Depth2 new terms: {len(depth2_new)}")
print(f"Total RAW unique terms: {len(all_terms)}")

# ---- 5) Sauvegardes
pd.Series(depth1_new if depth1_new else [""], name="term").to_csv("keywords_depth1_raw.csv", index=False)
pd.Series(depth2_new if depth2_new else [""], name="term").to_csv("keywords_depth2_raw.csv", index=False)
pd.Series(sorted(all_terms), name="term").to_csv("keywords_all_raw.csv", index=False)

print("\n✓ Saved: keywords_depth1_raw.csv, keywords_depth2_raw.csv, keywords_all_raw.csv")
print("="*70 + "\n")