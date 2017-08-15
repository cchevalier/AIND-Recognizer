from unittest import TestCase

from asl_data import AslDb
from my_model_selectors import (
    SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV,
)

FEATURES = ['right-y', 'right-x']

class TestSelectors(TestCase):
    def setUp(self):
        asl = AslDb()
        self.training = asl.build_training(FEATURES)
        self.sequences = self.training.get_all_sequences()
        self.xlengths = self.training.get_all_Xlengths()

    def test_select_constant_interface(self):
        model = SelectorConstant(self.sequences, self.xlengths, 'BUY', verbose=False).select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorConstant(self.sequences, self.xlengths, 'BOOK', verbose=False).select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_bic_interface(self):
        model = SelectorBIC(self.sequences, self.xlengths, 'FRANK', verbose=False).select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorBIC(self.sequences, self.xlengths, 'VEGETABLE', verbose=False).select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_cv_interface(self):
        model = SelectorCV(self.sequences, self.xlengths, 'JOHN', verbose=False).select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorCV(self.sequences, self.xlengths, 'CHICKEN', verbose=False).select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_dic_interface(self):
        model = SelectorDIC(self.sequences, self.xlengths, 'MARY', verbose=False).select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorDIC(self.sequences, self.xlengths, 'TOY', verbose=False).select()
        self.assertGreaterEqual(model.n_components, 2)
