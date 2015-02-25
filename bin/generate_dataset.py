import json
import numpy as np
import os
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

#####
# Setup instructions: Create data/ directory in the the project directory
# Download and place train.json and test.json inside it.
# I added data/ to .gitignore so anything in it won't be added to git
# Usage:
# python bin/generate_dataset.py data/ data/
#

inputdir = sys.argv[1]
outputdir = sys.argv[2]

f = open(os.path.join(inputdir, 'train.json'), 'r')
all_data = json.loads(f.read())
f.close()
f = open(os.path.join(inputdir, 'test.json'), 'r')
test_data = json.loads(f.read())
f.close()

def update_feature_name_to_idx(vectorizer, st_idx, prefix):
    feature_name_to_idx.update(
        dict([
            (prefix + x, st_idx + i)
            for i, x in enumerate(vectorizer.get_feature_names())
        ])
    )

# build vectorizers
feature_name_to_idx = {}
dict_vectorizer = DictVectorizer()
keys_to_dict_vectorize = [
#    'number_of_upvotes_of_request_at_retrieval',
#    'number_of_downvotes_of_request_at_retrieval',
#    'post_was_edited',
    'requester_account_age_in_days_at_request',
#    'requester_account_age_in_days_at_retrieval',
    'requester_days_since_first_post_on_raop_at_request',
#    'requester_days_since_first_post_on_raop_at_retrieval',
    'requester_number_of_comments_at_request',
#    'requester_number_of_comments_at_retrieval',
#    'requester_number_of_comments_in_raop_at_retrieval',
    'requester_number_of_comments_in_raop_at_request',                
    'requester_number_of_posts_at_request',
#    'requester_number_of_posts_at_retrieval',
    'requester_number_of_posts_on_raop_at_request',
#    'requester_number_of_posts_on_raop_at_retrieval',
    'requester_number_of_subreddits_at_request',
    'requester_upvotes_minus_downvotes_at_request',
#    'requester_upvotes_minus_downvotes_at_retrieval',
    'requester_upvotes_plus_downvotes_at_request',
#   'requester_upvotes_plus_downvotes_at_retrieval',
]
dict_vector_data = []
for data in all_data:
    dict_vector_data.append(dict([(x, data[x]) for x in keys_to_dict_vectorize]))
dict_vectorizer.fit(dict_vector_data)
update_feature_name_to_idx(dict_vectorizer, 0, "")

count_vectorizer = CountVectorizer(analyzer='word', min_df=20, stop_words='english')
rq_txt_key = 'request_text_edit_aware'
count_vectorizer.fit(
    [x[rq_txt_key] for x in all_data] +
    [x[rq_txt_key] for x in test_data]
)
update_feature_name_to_idx(
    count_vectorizer,
    len(feature_name_to_idx),
    "",
)

title_cnt_vectorizer = CountVectorizer(analyzer='word', min_df=3, stop_words='english')
title_cnt_vectorizer.fit(
    [x['request_title'] for x in all_data] + 
    [x['request_title'] for x in test_data]
)
update_feature_name_to_idx(
    title_cnt_vectorizer,
    len(feature_name_to_idx),
    "TITLE_",
)

subreddit_cnt_vectorizer = CountVectorizer(analyzer='word', min_df=0, stop_words='english')
subreddit_cnt_vectorizer.fit(
    [" ".join(x['requester_subreddits_at_request']) for x in all_data] +
    [" ".join(x['requester_subreddits_at_request']) for x in test_data]
)
update_feature_name_to_idx(
    subreddit_cnt_vectorizer,
    len(feature_name_to_idx),
    "SUBREDDIT_",
)

def build_features(data):
    dict_array = dict_vectorizer.transform(data)
    dict_array = np.array(dict_array.toarray())
    
    bow_array = count_vectorizer.transform([x[rq_txt_key] for x in data])
    bow_array = np.array(bow_array.toarray())

    title_bow_array = title_cnt_vectorizer.transform([x['request_title'] for x in data])
    title_bow_array = np.array(title_bow_array.toarray()) 

    # add boolean feature - giver_user_name_if_known
    bool_array = np.array(
        [x['giver_username_if_known'] != 'N/A' for x in data]
    )[:, np.newaxis]

    # subreddits features
    # not using subreddits for now -- adding a lot of dimensions
    subreddits_array = subreddit_cnt_vectorizer.transform(
        [" ".join(x['requester_subreddits_at_request']) for x in data]
    )
    subreddits_array = np.array(subreddits_array.toarray())
    print(dict_array.shape, bow_array.shape, title_bow_array.shape, bool_array.shape, subreddits_array.shape)

    return np.hstack([dict_array, bow_array, title_bow_array, bool_array])

def apply_transforms(features):
    # Apply transformations
    columns_to_log_transform = [
        'requester_number_of_comments_at_request',
        'requester_number_of_comments_in_raop_at_request',                
        'requester_number_of_posts_at_request',
        'requester_number_of_posts_on_raop_at_request',
    
        'requester_number_of_subreddits_at_request',

        'requester_upvotes_plus_downvotes_at_request',
    ]

    tr_features = features.copy()
    for colname in columns_to_log_transform:
        idx = feature_name_to_idx[colname]
        if any(x < 0. for x in tr_features[:, idx]):
            print (colname, tr_features[np.where(tr_features[:, idx] <  0),  idx])
            raise RuntimeError("xx")
    tr_features[:, idx] = np.log(tr_features[:, idx] + 1)
    return tr_features

def filepath(fname):
    return os.path.join(outputdir, fname)

tr_features = apply_transforms(build_features(all_data))
np.savetxt(filepath('train.csv'), tr_features, delimiter=",")
y = np.array([int(data['requester_received_pizza']) for data in all_data])
np.savetxt(filepath('y_train.csv'), y, delimiter=",")

# Generate samples for training
train_sample, test_sample, y_train_sample, y_test_sample = train_test_split(tr_features, y, test_size=0.2, random_state=40)
np.savetxt(filepath('train_sample.csv'), train_sample, delimiter=",")
np.savetxt(filepath('test_sample.csv'), test_sample, delimiter=",")
np.savetxt(filepath('y_train_sample.csv'), y_train_sample, delimiter=",")
np.savetxt(filepath('y_test_sample.csv'), y_test_sample, delimiter=",")

tr_features = apply_transforms(build_features(test_data))
np.savetxt(filepath('test.csv'), tr_features, delimiter=",")
