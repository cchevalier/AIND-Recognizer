import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self,
                 all_word_sequences: dict,
                 all_word_Xlengths: dict,
                 this_word: str,
                 n_constant=3,
                 min_n_components=2,
                 max_n_components=10,
                 random_state=14,
                 verbose=False):
        """ init """
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_model = None
        lowest_score = float("+inf")

        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                model = GaussianHMM(n_components=n,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)

                # BIC score = -2 * logL + p * logN
                logL = model.score(self.X, self.lengths)

                # p: nb of parameters  = n * n + 2 * n * d - 1
                #      where: n = n_components
                #             d = nb of features
                # see https://discussions.udacity.com/t/bayesian-information-criteria-equation/326887/2
                d = model.n_features
                p = n * n + 2 * n * d - 1

                # N: nb of data points
                # see https://discussions.udacity.com/t/number-of-data-points-bic-calculation/235294/4
                N = len(self.X)
                logN = np.log(N)

                score = -2.0 * logL + p * logN

                if self.verbose:
                    print("   model created for {} with {} states, BIC score {:.2f}".format(self.this_word, n, score))

            except ValueError:
                model = None
                score = float("+inf")

                if self.verbose:
                    print("   failure on {} with {} states".format(self.this_word, n))

            finally:
                if score < lowest_score:
                    best_model = model
                    lowest_score = score

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        best_model = None
        highest_score = float("-inf")

        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                model = GaussianHMM(n_components=n,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)

                # DIC score = logL(this_word) - mean(logL(other_words))
                logL = model.score(self.X, self.lengths)

                logL_others = []
                for word in self.words:
                    if word != self.this_word:
                        word_X, word_lengths = self.hwords[word]
                        logL_others.append(model.score(word_X, word_lengths))

                score = logL - np.mean(logL_others)

                if self.verbose:
                    print("   model created for {} with {} states, DIC score {:.2f}".format(self.this_word, n, score))

            except ValueError:
                model = None
                score = float("-inf")

                if self.verbose:
                    print("   failure on {} with {} states".format(self.this_word, n))

            finally:
                if score > highest_score:
                    best_model = model
                    highest_score = score

        return best_model


class SelectorCV(ModelSelector):
    """
    select best model based on average log Likelihood of cross-validation folds
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        best_model = None
        highest_score = float("-inf")

        n_splits = min(3, len(self.sequences))
        split_method = KFold(n_splits)

        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                cv_score = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                    cv_model = GaussianHMM(n_components=n,
                                           covariance_type="diag",
                                           n_iter=1000,
                                           random_state=self.random_state,
                                           verbose=False).fit(train_X, train_lengths)
                    cv_score.append(cv_model.score(test_X, test_lengths))

                score = np.mean(cv_score)

                if score > highest_score:
                    # Recompute model based on all data but keep score as mean(cv_score)
                    model = GaussianHMM(n_components=n,
                                        covariance_type="diag",
                                        n_iter=1000,
                                        random_state=self.random_state,
                                        verbose=False).fit(self.X, self.lengths)
                    # score = model.score(self.X, self.lengths)

                if self.verbose:
                    print("   model created for {} with {} states, score {:.2f} and {} cv splits".format(self.this_word,
                                                                                                         n, score,
                                                                                                         n_splits))

            except ValueError:
                model = None
                score = float("-inf")

                if self.verbose:
                    print("   failure on {} with {} states".format(self.this_word, n))

            finally:
                if score > highest_score:
                    best_model = model
                    highest_score = score

        return best_model
