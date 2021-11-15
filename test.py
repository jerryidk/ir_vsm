'''tests goes here'''

import numpy as np

import unittest
import buildindex
import vsm

def write_test_files():
    with open('./test/test_file_1', 'w') as w:
        w.write('MED-0 hello world hello\n')
        w.write('MED-1 hi hello are you world\n')
        w.write('MED-2 this is for test\n')

class TestIndex(unittest.TestCase):

    def test_simple_1(self):
        idx = buildindex.Index('./test/test_file_1')
        idx.writeToFile('./test/test_1_index')
        self.assertEqual(idx.getTFinC('hello'), 3, 'hello should be 3')
        self.assertEqual(idx.getTFinC('world'), 2, 'world should be 2')
        self.assertEqual(idx.getTFinC('test'), 1, 'test should be 1')
        self.assertEqual(idx.getTFinD('test', 'MED-1'), 0, 'test should be 0')
        self.assertEqual(idx.getDocNumContainT('world'), 2, 'docs containing world should be 2')
        self.assertEqual(idx.getDocNum(), 3, 'docs number should 3')

        model = vsm.VSM(idx, 'tf')
        self.assertEqual(len(model.word_index.keys()), 9, 'word_index should have 9 words')
        self.assertEqual(model.calculateTermFrequency().shape, (3,9), 'term frequency should be correct size')
        self.assertEqual(model.calculateDocumentFrequency().shape, (9,), 'document frequency should be correct size')
        self.assertEqual(model.calculateInverseDocumentFrequency().shape, (9,), 'inverse document frequency should be correct size')
        self.assertEqual(model.calculateTFIDF().shape, (3,9), 'tfidf should be correct size')
        np.testing.assert_array_equal(model.evaluateQuery('hello world', 3), ['MED-0', 'MED-1', 'MED-2'], 'query should return correct ranking for simply query')

if __name__ == '__main__':
    write_test_files()
    unittest.main()
