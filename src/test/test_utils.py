import unittest
import sys
import apido
import os
import datetime

sys.path.append(".")  # Adds the module to path


class TestSequences(unittest.TestCase):
    def test_save_history_as_csv_fail(self):
        self.assertRaises(
            AssertionError,
            lambda: apido.save_history_as_csv("save_path.notcsv", {}),
        )

    def test_save_history_as_csv(self):
        apido.save_history_as_csv(
            "_test_path.csv",
            {"loss": [1, 2, 3], "val_loss": [2, 4]},
            delimiter=",",
        )

        self.assertTrue(os.path.exists("./_test_path.csv"))

        scores = apido.read_csv("_test_path.csv", delimiter=",")
        os.unlink("_test_path.csv")
        self.assertEqual(scores["loss"], [1, 2])
        self.assertEqual(scores["val_loss"], [2, 4])

    def test_get_date_from_filename(self):
        time = datetime.datetime.now()
        filename = apido.get_checkpoint_name("test")
        retrieved_time = apido.get_date_from_filename(filename)

        self.assertLess(time - retrieved_time, datetime.timedelta(seconds=1))

    def test_parse_index(self):

        for i in range(100):
            self.assertListEqual(list(apido.parse_index(str(i))), [i])

        self.assertListEqual(list(apido.parse_index(":100")), list(range(100)))

        self.assertListEqual(
            list(apido.parse_index("10:100")), list(range(10, 100))
        )

        self.assertListEqual(
            list(apido.parse_index("10:100:10")), list(range(10, 100, 10))
        )

        self.assertListEqual(
            list(apido.parse_index(":100:10")), list(range(0, 100, 10))
        )

        indices = apido.parse_index("::10")
        self.assertEqual(next(indices), 0)
        self.assertEqual(next(indices), 10)

        indices = apido.parse_index("5::10")
        self.assertEqual(next(indices), 5)
        self.assertEqual(next(indices), 15)


if __name__ == "__main__":
    unittest.main()
