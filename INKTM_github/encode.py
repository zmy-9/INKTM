from scipy import sparse
from this_queue import OurQueue
from collections import defaultdict, Counter
from scipy.sparse import load_npz, save_npz, csr_matrix, find
from math import log
import pandas as pd
import numpy as np
import argparse
import time
import os


parser = argparse.ArgumentParser(description='Prepare datasets.')
parser.add_argument('--min_interactions', type=int, nargs='?', default=10)
parser.add_argument('--remove_nan_skills', type=bool, nargs='?', const=True, default=True)
options = parser.parse_args()

# read assist2009 dataset
assist09 = pd.read_csv("new_skill_builder.csv", encoding="latin1", index_col=False)

# Filter out users that have less than min_interactions interactions
assist09 = assist09.groupby("user_id").filter(lambda x: len(x) >= options.min_interactions)

# Remove NaN skills
if options.remove_nan_skills:
    assist09 = assist09[~assist09["skill_id"].isnull()]
else:
    assist09.loc[assist09["skill_id"].isnull(), "skill_id"] = -1

assist09 = assist09[['user_id', 'problem_id', 'skill_id', 'correct']]  # select part of features

assist09["item_id"] = np.unique(assist09["problem_id"], return_inverse=True)[1]
assist09["user_id"] = np.unique(assist09["user_id"], return_inverse=True)[1]

nb = Counter()
attempts = []
for user, item in zip(assist09["user_id"], assist09["item_id"]):
    attempts.append(nb[user, item])
    nb[user, item] += 1
assist09["attempts"] = attempts

assist09.reset_index(inplace=True, drop=True)

# Build q-matrix
listOfKC = []
for kc_raw in assist09["skill_id"].unique():
    for elt in str(kc_raw).split('_'):
        listOfKC.append(str(int(float(elt))))
listOfKC = np.unique(listOfKC)  # 将所有技能ID排序，删去重复的ID

dict1_kc = {}; dict2_kc = {}
for k, v in enumerate(listOfKC):  # 0 skill1, 1 skill2, ...
    dict1_kc[v] = k  # dict1_kc[skill1] = 0
    dict2_kc[k] = v  # dict2_kc[0] = skill1

# Build Q-matrix
Q_mat = np.zeros((len(assist09["item_id"].unique()), len(listOfKC)))  # num_question * num_skills
item_skill = np.array(assist09[["item_id", "skill_id"]])
for i in range(len(item_skill)):
    splitted_kc = str(item_skill[i, 1]).split('_')
    for kc in splitted_kc:
        Q_mat[item_skill[i,0], dict1_kc[str(int(float(kc)))]] = 1    # one-hot encoder
assist09.drop(['skill_id'], axis=1, inplace=True)
assist09 = assist09[assist09.correct.isin([0, 1])]  # Remove potential continuous outcomes
assist09['correct'] = assist09['correct'].astype(np.int32)  # Cast outcome as int32

# Save data
sparse.save_npz("q_mat_min10_builder.npz", sparse.csr_matrix(Q_mat))
assist09.to_csv("needed_min10_builder.csv", index=False)
print("proprecess successfully!")

dt = time.time()
full = assist09.copy()
nb_samples, _ = full.shape
shift_skills = 0
if full['user_id'].dtype == np.int64:  # We shift IDs to ensure that
    shift_items = 1 + full['user_id'].max()  # user/item/skill IDs are distinct
    full['item_id'] += shift_items  # encode user_ID、question_ID, and skill_ID
    shift_skills = int(1 + full['item_id'].max())

# Handle skills (either q-matrix, or skill_id, or skill_ids from 0)
q_mat = defaultdict(list)  # similar to dict
nb_skills = None
if 'skill_id' in full.columns:
    print('Found a column skill_id')
    full['skill_id'] += shift_skills
elif os.path.isfile('q_mat_min10.npz'):
    print('Found a q-matrix')
    q_matrix = load_npz('q_mat_min10.npz')
    _, nb_skills = q_matrix.shape  # _: num_question; nb_skills: num_skills
    rows_one, cols_one, _ = find(q_matrix)  # return the positions of non-0 elements
    for i, j in zip(rows_one, cols_one):
        q_mat[shift_items + i].append(shift_skills + j)  # q_mat[question]: skills[s1,s3,s4,s5]

full['i'] = range(nb_samples)
print('Loading data:', nb_samples, 'samples', time.time() - dt)
print(full.head())
all_values = {}

for col in {'user_id', 'item_id', 'skill_id'}:
    if col in full.columns:
        all_values[col] = full[col].dropna().unique()
    else:
        all_values['skill_id'] = list(range(shift_skills,
                                            shift_skills + nb_skills))

print("shiftitems:", shift_items)
print("shiftskills:", shift_skills)
print("nb_skills:", nb_skills)

conversion = {
    'user_id': 'user',
    'item_id': 'item',
    'skill_id': 'kc'
}

# Preprocess codes
dt = time.time()
print([value for field, key in conversion.items()
       for value in all_values[field]])  # [0,2,1,3,4,5,6,7,8,9]
codes = dict(zip([value for field, key in conversion.items()
                  for value in all_values[field]], range(1000000)))
print('Preprocess codes', time.time() - dt)
print(codes)

# Extra codes for counters within time windows (wins, attempts)
extra_codes = dict(zip([(field, value)
                        for value in all_values['skill_id']
                        for field in {'wins'}],
                       range(1000000)))
print(extra_codes)
extra_codes_fail = dict(zip([(field, value)
                             for value in all_values['skill_id']
                             for field in {'fails'}],
                            range(1000000)))
# print(extra_codes)
print(len(codes))
print(len(extra_codes))
print(len(extra_codes_fail))
print('Gather all', len(codes) + len(extra_codes) + len(extra_codes_fail), 'features')

convert = np.vectorize(codes.get)

for field, key in conversion.items():
    dt = time.time()
    if field != 'skill_id':  # Will not work because of potential NaN values
        full[key] = convert(full[field])
        print('Encode', key, time.time() - dt)

dt = time.time()
rows_one = list(range(nb_samples)) + list(range(nb_samples))  # User & Item
cols_one = list(full['user']) + list(full['item'])
data_one = [1] * (2 * nb_samples)
assert len(rows_one) == len(cols_one) == len(data_one)
print('Initialized', len(rows_one), 'entries', time.time() - dt)
rows_wins, rows_fails = [], []
cols_wins, cols_fails = [], []
data_wins, data_fails = [], []


def add_one(r, c, d):
    rows_one.append(r)
    cols_one.append(c)
    data_one.append(d)


def add_wins(r, c, d):
    rows_wins.append(r)
    cols_wins.append(c)
    data_wins.append(d)


def add_fails(r, c, d):
    rows_fails.append(r)
    cols_fails.append(c)
    data_fails.append(d)


def identity(x):
    return x

if options.tw or options.pfa:  # Build time windows features
    df = full
    if 'skill_id' in full.columns:
        df = df.dropna(subset=['skill_id'])
        df['skill_ids'] = df['skill_id'].astype(str)
    else:
        df['skill_ids'] = [None] * len(df)

    # Prepare counters for time windows
    q = defaultdict(lambda: OurQueue(only_forever=options.pfa))
    for i_sample, user, item_id, correct, skill_ids in zip(
            df['i'], df['user'], df['item_id'], df['correct'],
            df['skill_ids']):
        f = 1
        w = 1
        for skill_id in q_mat[item_id]:  # Fallback
            skill_id = int(skill_id)
            add_one(i_sample, codes[skill_id], 1)

            fail = q[user, skill_id].get_counters(f)
            add_fails(i_sample, extra_codes_fail['fails', skill_id], fail)

            win = q[user, skill_id, 'correct'].get_counters(w)
            add_wins(i_sample, extra_codes['wins', skill_id], win)
            if not correct:
                q[user, skill_id].push(f)
            else:
                q[user, skill_id, 'correct'].push(w)

print('Total', len(rows_one), 'entries')

X_one = csr_matrix((data_one, (rows_one, cols_one)))
X_wins = csr_matrix((data_wins, (rows_wins, cols_wins)))
X_fails = csr_matrix((data_fails, (rows_fails, cols_fails)))
print(X_one.todense())
print(X_wins.todense())
print(X_fails.todense())

y = np.array(full['correct'])
print('Into sparse matrix:', X_one.shape, y.shape)

save_npz('One-uiswf.npz', X_one)
save_npz('Wins-uiswf.npz', X_wins)
save_npz('Fails-uiswf.npz', X_fails)
np.save('Label-uiswf.npy', y)
print('Saving successfully!')

