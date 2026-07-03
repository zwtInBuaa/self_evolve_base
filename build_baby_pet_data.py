#!/usr/bin/env python3
"""
Batch download and process Baby and Pet datasets into data_cache format.
Same logic as build_beauty_data.py.
"""
import ast, csv, gzip, json, os, random, sys, urllib.request
from collections import defaultdict
from pathlib import Path

URL_PREFIX = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "_raw_data"
OUTPUT_BASE = SCRIPT_DIR / "data_cache"

RATING_THRESHOLD = 4
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 50
RANDOM_SEED = 42

DATASETS = {
    "Baby": {
        "ratings": "ratings_Baby.csv",
        "reviews": "reviews_Baby_5.json.gz",
        "meta": "meta_Baby.json.gz",
    },
    "Pet": {
        "ratings": "ratings_Pet_Supplies.csv",
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


def process_ratings(csv_path):
    user_events = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
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

    user_seqs = {}
    for user, events in user_events.items():
        events.sort(key=lambda x: x[0])
        seq = [item for _, item in events]
        if len(seq) < MIN_SEQ_LEN:
            continue
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
        user_seqs[user] = seq
    return user_seqs


def split_data(user_seqs):
    train_hist, train_time = {}, {}
    valid_hist, valid_time = {}, {}
    test_hist, test_time = {}, {}
    for user, seq in user_seqs.items():
        train_seq = seq[:-2]
        if len(train_seq) < 1:
            continue
        valid_seq = seq[:-1]
        test_seq_list = seq
        train_hist[user] = train_seq
        train_time[user] = [str(i) for i in range(len(train_seq))]
        valid_hist[user] = valid_seq
        valid_time[user] = [str(i) for i in range(len(valid_seq))]
        test_hist[user] = test_seq_list
        test_time[user] = [str(i) for i in range(len(test_seq_list))]
    return (train_hist, train_time), (valid_hist, valid_time), (test_hist, test_time)


def reindex(train_hist_raw, valid_hist_raw, test_hist_raw, train_time_raw, valid_time_raw, test_time_raw):
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

    def rh(d):
        return {str(user2id[u]): [item2id[it] for it in seq] for u, seq in d.items()}

    def rt(td, hd):
        return {str(user2id[u]): times for u, times in td.items() if u in hd}

    return (user2id, item2id, id2user, id2item,
            rh(train_hist_raw), rt(train_time_raw, train_hist_raw),
            rh(valid_hist_raw), rt(valid_time_raw, valid_hist_raw),
            rh(test_hist_raw), rt(test_time_raw, test_hist_raw))


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
    # user_sets.txt
    user_ids = sorted(train_hist.keys(), key=int)
    with open(output_dir / "user_sets.txt", "w", encoding="utf-8") as f:
        for uid in user_ids:
            f.write(f"{uid}\n")

    train_items = sum(len(s) for s in train_hist.values())
    print(f"    Users={len(user2id)}, Items={len(item2id)}, Train interactions={train_items}")
    print(f"    Meta items={len(meta)}, Review users={len(review_dict)}")


def process_dataset(name, files):
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    # 1. Download
    RAW_DIR.mkdir(exist_ok=True)
    for key, fname in files.items():
        download(URL_PREFIX + fname, RAW_DIR / fname)

    # 2. Process ratings
    user_seqs = process_ratings(RAW_DIR / files["ratings"])
    print(f"  Users after filtering: {len(user_seqs)}")

    # 3. Split
    (train_hist, train_time), (valid_hist, valid_time), (test_hist, test_time) = split_data(user_seqs)

    # 4. Reindex
    (user2id, item2id, id2user, id2item,
     train_hist, train_time, valid_hist, valid_time, test_hist, test_time) = reindex(
        train_hist, valid_hist, test_hist, train_time, valid_time, test_time)

    # 5. Metadata
    meta = process_metadata(RAW_DIR / files["meta"], item2id)

    # 6. Reviews
    review_dict, train_review_dict = process_reviews(RAW_DIR / files["reviews"], user2id, item2id, train_hist)

    # 7. Write
    output_dir = OUTPUT_BASE / name
    write_output(output_dir, train_hist, train_time, valid_hist, valid_time, test_hist, test_time,
                 user2id, item2id, id2user, id2item, review_dict, train_review_dict, meta)

    print(f"  Done! Output: {output_dir}")


def main():
    random.seed(RANDOM_SEED)
    for name, files in DATASETS.items():
        process_dataset(name, files)
    print("\nAll done!")


if __name__ == "__main__":
    main()