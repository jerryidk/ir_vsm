'''tests goes here'''

import unittest
import buildindex


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


if __name__ == '__main__':
    write_test_files()
    unittest.main()
