from glob import glob
import os.path


def get_env_name(fname):
    basename = os.path.basename(fname)
    env_name, ext = os.path.splitext(basename)
    assert ext == ".tsv"
    return env_name

def read_processed_data(fname):
    """Read processed data as a list of dictionaries.

    Reads from TSV lines with the following header line:
    sentence    label
    """
    examples = []
    with open(fname, encoding="utf-8") as f:
        for (i, line) in enumerate(f):
            if i == 0:
                continue
            text, label = line.split("\t")
            examples.append({'text': text, 'label': label})
    return examples

def get_train_examples(data_dir):
    """Load training examples from multiple environments."""
    train_envs = glob(os.path.join(data_dir, "train*.tsv"))
    print(f"loading train data from {len(train_envs)} environments")

    # return examples as dict; check that sizes match for training
    train_examples = {}
    prev_length = None
    for train_env in train_envs:
        env_name = get_env_name(train_env)
        examples = read_processed_data(train_env)
        if prev_length:
            assert len(examples) == prev_length, \
                f"data size between training environments differ"

        train_examples[env_name] = examples
        prev_length = len(examples)
    return train_examples


def get_test_examples(data_dir):
    """Load test examples from multiple environments."""
    test_envs = glob(os.path.join(data_dir, "test*.tsv"))
    print(f"loading test data from {len(test_envs)} environments")

    # test0: examples0, test1: examples1, test_ood: examples_ood
    test_examples = {}
    for test_env in test_envs:
        env_name = get_env_name(test_env)
        examples = read_processed_data(test_env)
        test_examples[env_name] = examples
    return test_examples