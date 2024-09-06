import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
import collections
CUDA = torch.cuda.is_available()


def show_game(original_word, guesses, obscured_words_seen):
    print('Hidden word was "{}"'.format(original_word))

    for i in range(len(guesses)):
        word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in obscured_words_seen[i].argmax(axis=1)])
        print('Guessed {} after seeing "{}"'.format(guesses[i], word_seen))

def list2tensor(arr):
    arr = np.array(arr)
    return torch.from_numpy(arr)


class Word2Batch:
    def __init__(self, model, word, lives=6):
        self.origin_word = word
        self.guessed_letter = set()  # each element should be a idx
        self.word_idx = [ord(i)-97 for i in word]
        self.remain_letters = set(self.word_idx)
        self.model = model
        self.lives_left = lives
        self.guessed_letter_each = []

        # the following is the dataset for variable to output
        self.obscured_word_seen = []  # n * 27, where n is the number of guesses
        self.prev_guessed = []  # n*26, where n is the number of guesses and each element is the normalized word idx
        self.correct_response = []  # this is the label, meaning self.prev_guess should be one of self.correct_response

    def encode_obscure_word(self):
        word = [i if i in self.guessed_letter else 26 for i in self.word_idx]
        obscured_word = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word):
            obscured_word[i, j] = 1
        return obscured_word

    def encode_prev_guess(self):
        guess = np.zeros(26, dtype=np.float32)
        for i in self.guessed_letter:
            guess[i] = 1.0
        return guess

    def encode_correct_response(self):
        response = np.zeros(26, dtype=np.float32)
        for i in self.remain_letters:
            response[i] = 1.0
        response /= response.sum()
        return response

    def game_mimic(self):
        obscured_words_seen = []
        prev_guess_seen = []
        correct_response_seen = []

        while self.lives_left > 0 and len(self.remain_letters) > 0:
            # store obscured word and previous guesses -- act as X for the label
            obscured_word = self.encode_obscure_word()
            prev_guess = self.encode_prev_guess()

            obscured_words_seen.append(obscured_word)
            prev_guess_seen.append(prev_guess)
            obscured_word = torch.from_numpy(obscured_word)
            prev_guess = torch.from_numpy(prev_guess)
            if CUDA:
                obscured_word = obscured_word.cuda()
                prev_guess = prev_guess.cuda()

            model.eval()
            guess = self.model(obscured_word, prev_guess)  # output of guess should be a 1 by 26 vector
            # Replace the elements at indices in guessed_letters with a very small value
            #for idx in self.guessed_letters:
            #    guess[0, idx] = -1e9  # Set it to a very small value (negative infinity)
            guess = torch.argmax(guess, dim=2).item()
            self.guessed_letter.add(guess)
            self.guessed_letter_each.append(chr(guess + 97))

            # store correct response -- act as label for the model
            correct_response = self.encode_correct_response()
            correct_response_seen.append(correct_response)

            # update letter remained and lives left
            if guess in self.remain_letters:  # only remove guess when the guess is correct
                self.remain_letters.remove(guess)

            if correct_response_seen[-1][guess] < 0.0000001:  # which means we made a wrong guess
                self.lives_left -= 1
        obscured_words_seen = list2tensor(obscured_words_seen)
        prev_guess_seen = list2tensor(prev_guess_seen)
        correct_response_seen = list2tensor(correct_response_seen)
        # correct_response_seen = correct_response_seen.long()
        return obscured_words_seen, prev_guess_seen, correct_response_seen

class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()

        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size

        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size))
            self.h = Variable(torch.zeros(state_size))
            if CUDA:
                self.c = self.c.cuda()
                self.h = self.h.cuda()

        self.h, self.c = self.lstm(x, (self.h, self.c))

        return self.h


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x


class RNN_model(nn.Module):
    def __init__(self, hidden_units=16, target_dim=26):
        super(RNN_model, self).__init__()

        # self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)#,padding_idx=0)

        self.lstm1 = StatefulLSTM(27, hidden_units)
        self.bn_lstm1 = nn.BatchNorm1d(hidden_units)
        self.dropout1 = LockedDropout()
        self.fc = nn.Linear(hidden_units + 26, target_dim)


        # self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()

    def forward(self, obscure_word, prev_guess, train=True):
        if len(obscure_word.size()) < 3:
            obscure_word = obscure_word.unsqueeze(0)
        if len(prev_guess.size()) < 2:
            prev_guess = prev_guess.unsqueeze(0)

        no_of_timesteps = obscure_word.shape[0]
        batch_size = obscure_word.shape[1]
        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):
            h = self.lstm1(obscure_word[i, :, :])
            h = self.bn_lstm1(h)
            h = self.dropout1(h, dropout=0.1, train=train)

            pool = nn.MaxPool1d(batch_size)
            h = h.permute(1, 0)  # (batch_size,features,time_steps)
            h = h.unsqueeze(0)
            out = pool(h)
            out = out.squeeze(2)
            curr_prev_guess = prev_guess[i, :]
            curr_prev_guess = curr_prev_guess.unsqueeze(0)
            out = torch.cat((out, curr_prev_guess), 1)
            out = self.fc(out)
            outputs.append(out)
        outputs = torch.stack(outputs)
        return outputs

def gen_n_gram(word, n):
    n_gram = []
    for i in range(n, len(word)+1):
        if word[i-n:i] not in n_gram:
            n_gram.append(word[i-n:i])
    return n_gram

def init_n_gram(n):
    n_gram = {}
    full_dictionary = ["apple", "hhh", "genereate", "google", "abc", "googla"]
    for word in full_dictionary:
        single_word_gram = gen_n_gram(word, n)
        print(word, single_word_gram)
        if len(word) not in n_gram:
            n_gram[len(word)] = single_word_gram
        else:
            n_gram[len(word)].extend(single_word_gram)
    print(n_gram)
    res = {}
    for key in n_gram.keys():
        res[key] = collections.Counter(n_gram[key])
    return res

# def show_game(original_word, guesses, obscured_words_seen):
#     print('Hidden word was "{}"'.format(original_word))
#     for i in range(len(guesses)):
#         word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in obscured_words_seen[i].argmax(axis=1)])
#         print('Guessed {} after seeing "{}"'.format(guesses[i], word_seen))


# def get_all_words(file_location):
#     with open(file_location, "r") as text_file:
#         all_words = text_file.read().splitlines()
#     return all_words


# # get data
# #root_path = os.getcwd()
# #file_name = "words_250000_train.txt"
# file_path = r"/Users/kunaalgautam17/Desktop/hangman/words_250000_train.txt"
# words = get_all_words(file_path)
# num_words = len(words)

# # define model
# model = RNN_model(target_dim=26, hidden_units=16)

# # define hyper parameter
# n_epoch = 2
# lr = 0.001
# record_step = 100  # output result every 100 words
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# loss_func = nn.BCEWithLogitsLoss()
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)

# # start training
# start_time = time.perf_counter()
# tot_sample = 0
# for n in range(n_epoch):
#     i = 0
    
#     while tot_sample < (n + 1) * num_words:
#         word = words[i]
#         #if len(word) == 1:
#         #    continue
#         i += 1
#         # generate data in a batch
#         new_batch = Word2Batch(word=word, model=model)
#         obscured_word, prev_guess, correct_response = new_batch.game_mimic()
#         if CUDA:
#             obscured_word = obscured_word.cuda()
#         optimizer.zero_grad()
#         predict = model(obscured_word, prev_guess)
#         correct_response = correct_response.unsqueeze(1)
#         loss = loss_func(predict, correct_response)
#         loss.backward()
#         optimizer.step()
#         # show loss
#         curr_time = time.perf_counter()
#         print("for word {}, the BCE loss is {:4f}, time used:{:4f}".format(word, loss.item(), curr_time - start_time))
#         # show guess status
#         if i % record_step == 0:
#             guesses = [chr(i+97) for i in torch.argmax(prev_guess, 1)]
#             show_game(word, guesses, obscured_word)
#         tot_sample += 1





