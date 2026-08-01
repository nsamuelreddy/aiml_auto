from sklearn.model_selection import train_test_split


def split_dataset(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=True
):

    if stratify:

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    else:

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state
        )

    return x_train, x_test, y_train, y_test


def split_report(
    x_train,
    x_test,
    y_train,
    y_test
):

    report = {

        "Training Samples": len(x_train),

        "Testing Samples": len(x_test),

        "Training Labels": len(y_train),

        "Testing Labels": len(y_test)

    }

    return report