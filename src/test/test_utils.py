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
            lambda: apido.save_history_as_csv(
                "save_path.notcsv", {"data": [1, 2, 3]}, {}
            ),
        )

    def test_save_history_as_csv(self):
        apido.save_history_as_csv(
            "_test_path.csv",
            {"loss": [1, 2, 3], "val_loss": [2, 4]},
            {"arg1": 1, "arg2": 2},
            delimiter=",",
        )

        self.assertTrue(os.path.exists("./_test_path.csv"))

        args, scores = apido.read_csv("_test_path.csv", delimiter=",")
        os.unlink("_test_path.csv")
        self.assertEqual(args["arg1"], "1")
        self.assertEqual(args["arg2"], "2")
        self.assertEqual(scores["loss"], [1, 2])
        self.assertEqual(scores["val_loss"], [2, 4])

    def test_get_date_from_filename(self):
        time = datetime.datetime.now()

        filename = apido.get_checkpoint_name("test", 32)
        print(filename)
        retrieved_time = apido.get_date_from_filename(filename)

        self.assertLess(time - retrieved_time, datetime.timedelta(seconds=1))


datetime.time

if __name__ == "__main__":
    unittest.main()
