import unittest
import numpy as np
try:
    import torch
except ImportError:
    pass

import nlpaug.util.selection.filtering as filtering


class TestFiltering(unittest.TestCase):
    def test_probability_larger(self):
        data = np.array([-10, -0.1, 0.0, 0, 1.5, 3.4])

        # ============== Without replace value
        # Filter positive number
        modified_data, idxes = filtering.filter_proba(data, 3.3)
        np.testing.assert_equal(modified_data, np.array([3.4]))
        np.testing.assert_equal(idxes, np.array([5]))
        # Filter zero
        modified_data, idxes = filtering.filter_proba(data, 0)
        np.testing.assert_equal(modified_data, np.array([0., 0., 1.5, 3.4]))
        np.testing.assert_equal(idxes, np.array([2, 3, 4, 5]))
        # Filter negative number
        modified_data, idxes = filtering.filter_proba(data, -5)
        np.testing.assert_equal(modified_data, np.array([-0.1, 0., 0., 1.5, 3.4]))
        np.testing.assert_equal(idxes, np.array([1, 2, 3, 4, 5]))

        # ============== With replace value
        # Filter positive number
        modified_data, idxes = filtering.filter_proba(data, 2, replace=-99)
        np.testing.assert_equal(modified_data[modified_data > 2], np.array([3.4]))
        np.testing.assert_equal(idxes, np.array([5]))
        # Filter zero
        modified_data, idxes = filtering.filter_proba(data, 0, replace=-99)
        np.testing.assert_equal(modified_data, np.array([-99, -99, -99, -99, 1.5, 3.4]))
        np.testing.assert_equal(idxes, np.array([4, 5]))
        # Filter negative number
        modified_data, idxes = filtering.filter_proba(data, -5, replace=-99)
        np.testing.assert_equal(modified_data, np.array([-99, -0.1, 0., 0., 1.5, 3.4]))
        np.testing.assert_equal(idxes, np.array([1, 2, 3, 4, 5]))

    def test_probability_smaller(self):
        data = np.array([-10, -0.1, 0.0, 0, 1.5, 3.4])

        # ============== Without replace value
        # Filter positive number
        modified_data, idxes = filtering.filter_proba(data, 3.3, above=False)
        np.testing.assert_equal(modified_data, np.array([-10, -0.1, 0., 0., 1.5]))
        np.testing.assert_equal(idxes, np.array([0, 1, 2, 3, 4]))
        # Filter zero
        modified_data, idxes = filtering.filter_proba(data, 0, above=False)
        np.testing.assert_equal(modified_data, np.array([-10, -0.1, 0, 0]))
        np.testing.assert_equal(idxes, np.array([0, 1, 2, 3]))
        # # Filter negative number
        modified_data, idxes = filtering.filter_proba(data, -5, above=False)
        np.testing.assert_equal(modified_data, np.array([-10]))
        np.testing.assert_equal(idxes, np.array([0]))

        # ============== With replace value
        # Filter positive number
        modified_data, idxes = filtering.filter_proba(data, 2, above=False, replace=-99)
        np.testing.assert_equal(modified_data[modified_data > -99], np.array([-10, -0.1, 0., 0., 1.5]))
        np.testing.assert_equal(idxes, np.array([0, 1,2, 3, 4]))
        # # Filter zero
        modified_data, idxes = filtering.filter_proba(data, 0, above=False, replace=-99)
        np.testing.assert_equal(modified_data, np.array([-10, -0.1, -99, -99, -99, -99]))
        np.testing.assert_equal(idxes, np.array([0, 1]))
        # Filter negative number
        modified_data, idxes = filtering.filter_proba(data, -5, above=False, replace=-99)
        np.testing.assert_equal(modified_data, np.array([-10, -99, -99, -99, -99, -99]))
        np.testing.assert_equal(idxes, np.array([0]))

    def test_top_n_numpy(self):
        data = np.array([-10, -0.1, 0.0, 0, 3.4, 1.5])

        # ============== Without replace value, descending
        # Top 1
        modified_data, idxes = filtering.filter_top_k(data, 1)
        np.testing.assert_equal(modified_data, np.array([3.4]))
        np.testing.assert_equal(idxes, np.array([4]))
        # Top 2
        modified_data, idxes = filtering.filter_top_k(data, 2)
        np.testing.assert_equal(modified_data, np.array([3.4, 1.5]))
        np.testing.assert_equal(idxes, np.array([4, 5]))

        # ============== Without replace value, ascending
        # Top 1
        modified_data, idxes = filtering.filter_top_k(data, 1, ascending=True)
        np.testing.assert_equal(modified_data, np.array([3.4]))
        np.testing.assert_equal(idxes, np.array([4]))
        # Top 2
        modified_data, idxes = filtering.filter_top_k(data, 2, ascending=True)
        np.testing.assert_equal(modified_data, np.array([1.5, 3.4]))
        np.testing.assert_equal(idxes, np.array([5, 4]))
        # FIXME: Not yet handle
        # # Top same length
        # modified_data, idxes = filter_top_n(data, len(data))
        # np.testing.assert_equal(modified_data, np.array([-10, -0.1, 0., 0., 1.5, 3.4]))
        # np.testing.assert_equal(idxes, np.array([5, 4, 3, 2, 1, 0]))
        # FIXME: Not yet handle
        # # Top 100
        # modified_data = filtering.top_n(data, 100)
        # self.assert_lists(modified_data, data)

        # ============== With replace value
        modified_data, idxes = filtering.filter_top_k(data, 1, replace=-99)
        np.testing.assert_equal(modified_data, np.array([-99, -99, -99, -99, 3.4, -99]))
        np.testing.assert_equal(idxes, np.array([4]))
        modified_data, idxes = filtering.filter_top_k(data, 2, replace=-99)
        np.testing.assert_equal(modified_data, np.array([-99, -99, -99, -99, 3.4, 1.5]))
        np.testing.assert_equal(idxes, np.array([5, 4]))

    def test_top_n_pytorch(self):
        data = np.array([-10, -0.1, 0.0, 0, 3.4, 1.5])
        data = torch.tensor(data, dtype=torch.float)

        # ============== Without replace value, deascending
        # Top 1
        modified_data, idxes = filtering.filter_top_k(data, 1)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        np.testing.assert_equal(modified_data, np.array([3.4], dtype=np.float32))
        np.testing.assert_equal(idxes, np.array([4]))
        # Top 2
        modified_data, idxes = filtering.filter_top_k(data, 2)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        np.testing.assert_equal(modified_data, np.array([3.4, 1.5], dtype=np.float32))
        np.testing.assert_equal(idxes, np.array([4, 5]))

        # ============== Without replace value, ascending
        # Top 1
        modified_data, idxes = filtering.filter_top_k(data, 1, ascending=True)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        np.testing.assert_equal(modified_data, np.array([3.4], dtype=np.float32))
        np.testing.assert_equal(idxes, np.array([4]))
        # Top 2
        modified_data, idxes = filtering.filter_top_k(data, 2, ascending=True)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        np.testing.assert_equal(modified_data, np.array([1.5, 3.4], dtype=np.float32))
        np.testing.assert_equal(idxes, np.array([5, 4]))

        # ============== With replace value
        # Top 1
        modified_data, idxes = filtering.filter_top_k(data, 1, replace=-99)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        np.testing.assert_equal(modified_data, np.array([-99., -99., -99., -99., 3.4, -99.], dtype=np.float32))
        np.testing.assert_equal(idxes, np.array([4]))
        # Top 2
        modified_data, idxes = filtering.filter_top_k(data, 2, replace=-99)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        np.testing.assert_equal(modified_data, np.array([-99, -99, -99, -99, 3.4, 1.5], dtype=np.float32))
        np.testing.assert_equal(idxes, np.array([4, 5]))

    def test_cum_proba(self):
        data = torch.tensor([-9.2171, -18.5356, -18.8203, -10.8368, -13.3220, -11.5886])

        # ============== Without replace value
        modified_data, idxes = filtering.nucleus_sampling(data, 0.95, replace=None)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        expected_data = np.array([-9.2171, -10.8368], dtype=np.float32)
        np.testing.assert_equal(modified_data, expected_data)
        np.testing.assert_equal(idxes, np.array([0, 3]))

        modified_data, idxes = filtering.nucleus_sampling(data, 0.95, above=False, replace=None)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        expected_data = np.array([-11.5886, -13.3220, -18.5356, -18.8203], dtype=np.float32)
        np.testing.assert_equal(modified_data, expected_data)
        np.testing.assert_equal(idxes, np.array([5, 4, 1, 2]))

        # ============== With replace value
        modified_data, idxes = filtering.nucleus_sampling(data, 0.95)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        expected_data = np.array([-9.2171, -10.8368, 0.0000, 0.0000, 0.0000, 0.0000], dtype=np.float32)
        np.testing.assert_equal(modified_data, expected_data)
        np.testing.assert_equal(idxes, np.array([0, 3]))

        modified_data, idxes = filtering.nucleus_sampling(data, 0.95, above=False)
        modified_data = modified_data.data.numpy()
        idxes = idxes.data.numpy()
        expected_data = np.array([0.0000, 0.0000, -11.5886, -13.3220, -18.5356, -18.8203], dtype=np.float32)
        np.testing.assert_equal(modified_data, expected_data)
        np.testing.assert_equal(idxes, np.array([5, 4, 1, 2]))
