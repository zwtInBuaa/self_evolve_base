#!/usr/bin/env python3
"""
Download and process Amazon Beauty dataset into the same format as data_cache/CDs_and_Vinyl.

Output: data_cache/Beauty/
    train.json, valid.json, test.json,
    user2id.json, item2id.json, id2user.json, id2item.json,
    review.json, train_review.json, meta.json, user_sets.txt

Usage:
    python build_beauty_data.py
"""

import os, sys, gzip, json, csv, random, urllib.request
from collections import defaultdict
from pathlib import Path

# ===== Config =====
CATEGORY = "Beauty"
RATINGS_FILE = "ratings_Beauty.csv"
REVIEWS_FILE = "reviews_Beauty_5.json.gz"
META_FILE = "meta_Beauty.json.gz"
URL_PREFIX = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parent / "_raw_data"
OUTPUT_DIR = SCRIPT_DIR / "data_cache" / CATEGORY

RATING_THRESHOLD = 4
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 50
RANDOM_SEED = 42

# ===== Download =====
def download(url, dest):
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  downloading {dest.name} ({url})...")
    urllib.request.urlretrieve(url, dest)
    print(f"  done: {dest.name}")

def download_all():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("Step 1: Downloading Beauty dataset files...")
    print("=" * 60)
    download(URL_PREFIX + RATINGS_FILE, DATA_DIR / RATINGS_FILE)
    download(URL_PREFIX + REVIEWS_FILE, DATA_DIR / REVIEWS_FILE)
    download(URL_PREFIX + META_FILE, DATA_DIR / META_FILE)
    print()

# ===== Process Ratings → sequences =====
def process_ratings():
    print("=" * 60)
    print("Step 2: Processing ratings → user sequences...")
    print("=" * 60)

    csv_path = DATA_DIR / RATINGS_FILE
    user_events = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 4:
                continue
            user, item, rating_str, ts_str = row
            try:
                rating = float(rating_str)
                ts = int(ts_str)
            except (ValueError, TypeError):
                continue
            if rating < RATING_THRESHOLD:
                continue
            user_events[str(user)].append((ts, str(item)))

    # Sort by timestamp, keep last MAX_SEQ_LEN items
    user_seqs = {}
    for user, events in user_events.items():
        events.sort(key=lambda x: x[0])
        seq = [item for _, item in events]
        if len(seq) < MIN_SEQ_LEN:
            continue
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
        user_seqs[user] = seq

    print(f"  Users after filtering: {len(user_seqs)}")
    return user_seqs

# ===== Train/Valid/Test split =====
def split_data(user_seqs):
    print("=" * 60)
    print("Step 3: Train/Valid/Test split...")
    print("=" * 60)

    train_history, train_time = {}, {}
    valid_history, valid_time = {}, {}
    test_history, test_time = {}, {}

    for user, seq in user_seqs.items():
        # last = test, second-to-last = valid, rest = train
        train_seq = seq[:-2]
        valid_seq = seq[:-1]  # includes train + valid item
        test_seq = seq         # full sequence

        if len(train_seq) < 1:
            continue

        train_history[user] = train_seq
        train_time[user] = [str(i) for i in range(len(train_seq))]

        valid_history[user] = valid_seq
        valid_time[user] = [str(i) for i in range(len(valid_seq))]

        test_history[user] = test_seq
        test_time[user] = [str(i) for i in range(len(test_seq))]

    print(f"  Users: {len(train_history)}")
    return (train_history, train_time), (valid_history, valid_time), (test_history, test_time)

# ===== Reindex users and items =====
def reindex(train_hist_raw, valid_hist_raw, test_hist_raw,
             train_time_raw, valid_time_raw, test_time_raw):
    print("=" * 60)
    print("Step 4: Reindexing users and items...")
    print("=" * 60)

    all_users = sorted(set(train_hist_raw.keys()))
    all_items_set = set()
    for seq in train_hist_raw.values():
        all_items_set.update(seq)
    for seq in valid_hist_raw.values():
        all_items_set.update(seq)
    for seq in test_hist_raw.values():
        all_items_set.update(seq)
    all_items = sorted(all_items_set)

    user2id = {u: i + 1 for i, u in enumerate(all_users)}
    item2id = {it: i + 1 for i, it in enumerate(all_items)}
    id2user = {v: k for k, v in user2id.items()}
    id2item = {v: k for k, v in item2id.items()}

    def reindex_history(hist_dict):
        return {str(user2id[u]): [item2id[it] for it in seq]
                for u, seq in hist_dict.items()}

    def reindex_time(time_dict, hist_dict):
        return {str(user2id[u]): times
                for u, times in time_dict.items() if u in hist_dict}

    train_hist = reindex_history(train_hist_raw)
    valid_hist = reindex_history(valid_hist_raw)
    test_hist = reindex_history(test_hist_raw)

    train_time2 = reindex_time(train_time_raw, train_hist_raw)
    valid_time2 = reindex_time(valid_time_raw, valid_hist_raw)
    test_time2 = reindex_time(test_time_raw, test_hist_raw)

    print(f"  Users: {len(user2id)}, Items: {len(item2id)}")
    return (user2id, item2id, id2user, id2item,
            train_hist, train_time2, valid_hist, valid_time2, test_hist, test_time2)

# ===== Process metadata =====
def process_metadata(item2id):
    print("=" * 60)
    print("Step 5: Processing item metadata...")
    print("=" * 60)

    import ast
    meta_path = DATA_DIR / META_FILE
    meta = {}
    raw_count = 0
    matched_count = 0

    with gzip.open(meta_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                # Meta file is Python dict format (single quotes), not JSON
                item = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue

            asin = item.get("asin", "")
            if asin not in item2id:
                continue

            internal_id = str(item2id[asin])
            meta[internal_id] = {
                "title": str(item.get("title", "")),
                "average_rating": 0.0,
                "rating_number": 0,
                "price": "",
                "store": "",
                "categories": _flatten_categories(item.get("categories", [])),
            }
            matched_count += 1

    print(f"  Raw items: {raw_count}, Matched: {matched_count}/{len(item2id)}")
    return meta

def _flatten_categories(cats):
    """Flatten nested category list into a flat list of strings."""
    result = []
    if isinstance(cats, list):
        for c in cats:
            if isinstance(c, list):
                result.extend([str(x) for x in c])
            else:
                result.append(str(c))
    return result

# ===== Process reviews =====
def process_reviews(user2id, item2id, train_hist):
    print("=" * 60)
    print("Step 6: Processing reviews...")
    print("=" * 60)

    reviews_path = DATA_DIR / REVIEWS_FILE
    review = defaultdict(dict)
    train_review = defaultdict(dict)
    raw_count = 0
    matched_count = 0

    # Build set of training (user, item) pairs
    train_pairs = set()
    for uid, seq in train_hist.items():
        for iid in seq:
            train_pairs.add((str(uid), str(iid)))

    with gzip.open(reviews_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw_count += 1
            try:
                rev = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            reviewer = rev.get("reviewerID", "")
            asin = rev.get("asin", "")

            if reviewer not in user2id or asin not in item2id:
                continue

            uid = str(user2id[reviewer])
            iid = str(item2id[asin])

            review[uid][iid] = {
                "rating": float(rev.get("overall", rev.get("rating", 0)) or 0),
                "title": str(rev.get("summary", rev.get("title", "")) or ""),
                "text": str(rev.get("reviewText", rev.get("text", "")) or ""),
            }

            if (uid, iid) in train_pairs:
                train_review[uid][iid] = review[uid][iid]

            matched_count += 1

    # Convert to regular dicts for JSON serialization
    review_dict = {k: dict(v) for k, v in review.items()}
    train_review_dict = {k: dict(v) for k, v in train_review.items()}

    print(f"  Raw reviews: {raw_count}, Matched: {matched_count}")
    print(f"  Review users: {len(review_dict)}, Train review users: {len(train_review_dict)}")
    return review_dict, train_review_dict

# ===== Write output =====
def write_output(train_hist, train_time2, valid_hist, valid_time2, test_hist, test_time2,
                 user2id, item2id, id2user, id2item,
                 review_dict, train_review_dict, meta):
    print("=" * 60)
    print("Step 7: Writing output files...")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON files
    json_files = {
        "train.json": {"History": train_hist, "Time": train_time2},
        "valid.json": {"History": valid_hist, "Time": valid_time2},
        "test.json": {"History": test_hist, "Time": test_time2},
        "user2id.json": user2id,
        "item2id.json": item2id,
        "id2user.json": id2user,
        "id2item.json": id2item,
        "review.json": review_dict,
        "train_review.json": train_review_dict,
        "meta.json": meta,
    }

    for fname, data in json_files.items():
        path = OUTPUT_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  {fname} ({len(data)} entries)")

    # user_sets.txt
    user_sets_path = OUTPUT_DIR / "user_sets.txt"
    user_ids = sorted(train_hist.keys(), key=int)
    with open(user_sets_path, "w", encoding="utf-8") as f:
        for uid in user_ids:
            f.write(f"{uid}\n")
    print(f"  user_sets.txt ({len(user_ids)} lines)")

    # Stats
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    train_items = sum(len(s) for s in train_hist.values())
    valid_items = sum(len(s) for s in valid_hist.values())
    test_items = sum(len(s) for s in test_hist.values())
    print(f"  Users: {len(user2id)}")
    print(f"  Items: {len(item2id)}")
    print(f"  Train interactions: {train_items}")
    print(f"  Valid interactions: {valid_items}")
    print(f"  Test interactions:  {test_items}")
    print(f"  Reviews: {sum(len(v) for v in review_dict.values())}")
    print(f"  Meta items: {len(meta)}")
    print(f"\nOutput: {OUTPUT_DIR}")

# ===== Main =====
def main():
    random.seed(RANDOM_SEED)

    # 1. Download
    download_all()

    # 2. Process ratings
    user_seqs = process_ratings()

    # 3. Split
    (train_hist, train_time), (valid_hist, valid_time), (test_hist, test_time) = split_data(user_seqs)

    # 4. Reindex
    (user2id, item2id, id2user, id2item,
     train_hist, train_time, valid_hist, valid_time, test_hist, test_time) = reindex(
        train_hist, train_time, valid_hist, valid_time, test_hist, test_time)

    # 5. Metadata
    meta = process_metadata(item2id)

    # 6. Reviews
    review_dict, train_review_dict = process_reviews(user2id, item2id, train_hist)

    # 7. Write
    write_output(train_hist, train_time, valid_hist, valid_time, test_hist, test_time,
                 user2id, item2id, id2user, id2item,
                 review_dict, train_review_dict, meta)

    print("\nDone! Ready to use with self_evolverec.")

if __name__ == "__main__":
    main()