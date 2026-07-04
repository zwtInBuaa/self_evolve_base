#!/usr/bin/env python3
"""
Rebuild Beauty/Baby/Pet data from 5-core reviews (matching VirtualMLE processing).
Uses reviews_*_5.json.gz as the primary data source, NOT ratings.csv.
"""
import ast, gzip, json, os, random, urllib.request
from collections import defaultdict
from pathlib import Path

URL_PREFIX = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "_raw_data"
OUTPUT_BASE = SCRIPT_DIR / "data_cache"

MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 50
RANDOM_SEED = 42

DATASETS = {
    "Beauty": {
        "reviews": "reviews_Beauty_5.json.gz",
        "meta": "meta_Beauty.json.gz",
    },
    "Baby": {
        "reviews": "reviews_Baby_5.json.gz",
        "meta": "meta_Baby.json.gz",
    },
    "Pet": {
        "reviews": "reviews_Pet_Supplies_5.json.gz",
        "meta": "meta_Pet_Supplies.json.gz",
    },
}


def download(url, dest):
    if dest.exists():
        print(f"  [skip] {dest.name}")
        return
    print(f"  downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)


def load_interactions_from_5core(json_path):
    """Load interactions from 5-core reviews JSON (same as VirtualMLE)."""
    user_to_events = defaultdict(list)
    raw_items = set()
    raw_interactions = 0

    opener = gzip.open if str(json_path).endswith('.gz') else open
    mode = "rt" if str(json_path).endswith('.gz') else "r"

    with opener(json_path, mode, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            user = obj.get("reviewerID")
            item = obj.get("asin")
            timestamp = obj.get("unixReviewTime")
            if user is None or item is None or timestamp is None:
                continue
            try:
                timestamp = int(timestamp)
            except (TypeError, ValueError):
                continue

            user_to_events[str(user)].append((timestamp, line_no, str(item)))
            raw_items.add(str(item))
            raw_interactions += 1

    return user_to_events, len(raw_items), raw_interactions


def build_sequences(user_to_events, min_len, max_len):
    """Build item sequences, filter by length, truncate (same as VirtualMLE)."""
    user_to_seq = {}
    removed_short = 0
    for user, events in user_to_events.items():
        events.sort(key=lambda x: (x[0], x[1]))
        seq = [item for _, _, item in events]
        if len(seq) < min_len:
            removed_short += 1
            continue
        if len(seq) > max_len:
            seq = seq[-max_len:]
        user_to_seq[user] = seq
    return user_to_seq


def reindex_and_split(user_to_seq):
    """Reindex and split into train/valid/test (leave-one-out)."""
    all_users = sorted(user_to_seq.keys())
    all_items = sorted({item for seq in user_to_seq.values() for item in seq})

    user2id = {u: i + 1 for i, u in enumerate(all_users)}
    item2id = {it: i + 1 for i, it in enumerate(all_items)}
    id2user = {v: k for k, v in user2id.items()}
    id2item = {v: k for k, v in item2id.items()}

    train_hist, train_time = {}, {}
    valid_hist, valid_time = {}, {}
    test_hist, test_time = {}, {}

    for raw_user in all_users:
        uid = str(user2id[raw_user])
        seq = [item2id[it] for it in user_to_seq[raw_user]]

        if len(seq) < 3:
            continue

        train_seq = seq[:-2]
        valid_seq = seq[:-1]
        test_seq_list = seq

        if len(train_seq) < 1:
            continue

        train_hist[uid] = train_seq
        train_time[uid] = [str(i) for i in range(len(train_seq))]
        valid_hist[uid] = valid_seq
        valid_time[uid] = [str(i) for i in range(len(valid_seq))]
        test_hist[uid] = test_seq_list
        test_time[uid] = [str(i) for i in range(len(test_seq_list))]

    return (user2id, item2id, id2user, id2item,
            train_hist, train_time, valid_hist, valid_time, test_hist, test_time)


def process_metadata(meta_path, item2id):
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
                item = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            asin = item.get("asin", "")
            if asin not in item2id:
                continue
            cats = item.get("categories", [])
            flat_cats = []
            if isinstance(cats, list):
                for c in cats:
                    if isinstance(c, list):
                        flat_cats.extend([str(x) for x in c])
                    else:
                        flat_cats.append(str(c))
            internal_id = str(item2id[asin])
            meta[internal_id] = {
                "title": str(item.get("title", "")),
                "average_rating": 0.0,
                "rating_number": 0,
                "price": "",
                "store": "",
                "categories": flat_cats,
            }
            matched_count += 1
    print(f"    Meta: raw={raw_count}, matched={matched_count}/{len(item2id)}")
    return meta


def process_reviews(reviews_path, user2id, item2id, train_hist):
    review = defaultdict(dict)
    train_review = defaultdict(dict)
    train_pairs = set()
    for uid, seq in train_hist.items():
        for iid in seq:
            train_pairs.add((str(uid), str(iid)))

    raw_count = 0
    matched_count = 0
    opener = gzip.open if str(reviews_path).endswith('.gz') else open
    mode = "rt" if str(reviews_path).endswith('.gz') else "r"
    with opener(reviews_path, mode, encoding="utf-8", errors="replace") as f:
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
                "rating": float(rev.get("overall", 0) or 0),
                "title": str(rev.get("summary", "") or ""),
                "text": str(rev.get("reviewText", "") or ""),
            }
            if (uid, iid) in train_pairs:
                train_review[uid][iid] = review[uid][iid]
            matched_count += 1

    review_dict = {k: dict(v) for k, v in review.items()}
    train_review_dict = {k: dict(v) for k, v in train_review.items()}
    print(f"    Reviews: raw={raw_count}, matched={matched_count}, users={len(review_dict)}")
    return review_dict, train_review_dict


def write_output(output_dir, train_hist, train_time, valid_hist, valid_time, test_hist, test_time,
                 user2id, item2id, id2user, id2item, review_dict, train_review_dict, meta):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "train.json": {"History": train_hist, "Time": train_time},
        "valid.json": {"History": valid_hist, "Time": valid_time},
        "test.json": {"History": test_hist, "Time": test_time},
        "user2id.json": user2id, "item2id.json": item2id,
        "id2user.json": id2user, "id2item.json": id2item,
        "review.json": review_dict, "train_review.json": train_review_dict,
        "meta.json": meta,
    }
    for fname, data in files.items():
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    user_ids = sorted(train_hist.keys(), key=int)
    with open(output_dir / "user_sets.txt", "w", encoding="utf-8") as f:
        for uid in user_ids:
            f.write(f"{uid}\n")

    train_items = sum(len(s) for s in train_hist.values())
    print(f"    Users={len(user2id)}, Items={len(item2id)}, Train interactions={train_items}")


def process_dataset(name, files):
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    RAW_DIR.mkdir(exist_ok=True)
    for key, fname in files.items():
        download(URL_PREFIX + fname, RAW_DIR / fname)

    # 1. Load from 5-core reviews (same as VirtualMLE)
    user_to_events, raw_items, raw_interactions = load_interactions_from_5core(RAW_DIR / files["reviews"])
    print(f"  Raw: {len(user_to_events)} users, {raw_items} items, {raw_interactions} interactions")

    # 2. Build sequences
    user_to_seq = build_sequences(user_to_events, MIN_SEQ_LEN, MAX_SEQ_LEN)
    print(f"  After filtering: {len(user_to_seq)} users")

    # 3. Reindex and split
    (user2id, item2id, id2user, id2item,
     train_hist, train_time, valid_hist, valid_time, test_hist, test_time) = reindex_and_split(user_to_seq)

    # 4. Metadata
    meta = process_metadata(RAW_DIR / files["meta"], item2id)

    # 5. Reviews
    review_dict, train_review_dict = process_reviews(RAW_DIR / files["reviews"], user2id, item2id, train_hist)

    # 6. Write
    output_dir = OUTPUT_BASE / name
    write_output(output_dir, train_hist, train_time, valid_hist, valid_time, test_hist, test_time,
                 user2id, item2id, id2user, id2item, review_dict, train_review_dict, meta)


def main():
    random.seed(RANDOM_SEED)
    for name, files in DATASETS.items():
        process_dataset(name, files)
    print("\nAll done! Rebuilt from 5-core reviews.")


if __name__ == "__main__":
    main()