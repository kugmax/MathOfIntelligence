import pandas as pd

df = pd.read_csv(filepath_or_buffer="spam.csv", encoding='cp1252')
df.columns = ['v1', 'v2', 'none1', 'none2', 'none3']
df.dropna(how='all', inplace=True)
df = df.drop(['none1', 'none2', 'none3'], axis=1)

# print(df)

train_spam_words = {}
train_not_spam_words = {}

num_spam_examples = 0
num_total_examples = 0

num_spam_words = 0
num_not_spam_words = 0
num_total_words = 0

spam_probability = 0
not_spam_probability = 0

alpha = 1


def process_email(spam, body):
    global num_spam_words, num_not_spam_words
    global num_total_words

    for word in body.split():
        num_total_words += 1

        if spam == 'spam':
            train_spam_words[word] = train_spam_words.get(word, 0) + 1
            num_spam_words += 1
        else:
            train_not_spam_words[word] = train_not_spam_words.get(word, 0) + 1
            num_not_spam_words += 1


def train():
    global num_spam_examples, num_total_examples
    global spam_probability, not_spam_probability

    for email in df.values:
        num_total_examples += 1
        if email[0] == 'spam':
            num_spam_examples += 1

        process_email(email[0], email[1])

        spam_probability = num_spam_examples / float(num_total_examples)
        not_spam_probability = (num_total_examples - num_spam_examples) / float(num_total_examples)


def conditional_word(word, spam):
    if spam == 'spam':
        return (train_spam_words.get(word, 0) + alpha) / (num_spam_words + alpha) * num_total_words

    return (train_not_spam_words.get(word, 0) + alpha) / (num_not_spam_words + alpha) * num_total_words


def conditional_email(email, spam):
    result = 1.0
    for word in email.split():

        print(word, spam, conditional_word(word, spam))

        result *= conditional_word(word, spam)
    return result


def classify(email):
    is_spam = spam_probability * conditional_email(email, 'spam')
    is_not_spam = not_spam_probability * conditional_email(email, 'ham')

    print('---Is spam    : ', is_spam)
    print('---Is not spam: ', is_not_spam)

    return is_spam > is_not_spam


def main():
    train()

    # email = 'I HAVE A DATE ON SUNDAY WITH WILL!!'
    # email = 'Had your contract mobile 11 Mnths?'
    # email = 'REMINDER FROM O2: To get 2.50 pounds free'
    email = 'Lol you must buy that!!! winner'

    print('---So result is:', classify(email))


if __name__ == "__main__":
    main()

