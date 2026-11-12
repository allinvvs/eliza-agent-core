import os
import sys
import tempfile
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.memory import MemoryManager
from plugins import pc_monitor


class TestElizaCore(unittest.TestCase):
    def test_memory_idf_and_ring_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_path = os.path.join(tmpdir, "history.json")
            diary_path = os.path.join(tmpdir, "diary.txt")
            mem = MemoryManager(hist_path, diary_path, max_window=4)

            mem.write_to_diary("Настройка CUDA ускорения для видеокарты NVIDIA GTX 1650 прошла успешно.")
            mem.write_to_diary("Адаптивный сэмплер регулирует min_p и штраф за повторы.")

            res = mem.retrieve_crystals("Как работает видеокарта GTX 1650?", top_n=1)
            self.assertIn("GTX 1650", res)

    def test_pc_monitor_plugin(self):
        output = pc_monitor.run("железо")
        self.assertIn("ОС:", output)
        self.assertIn("Диск свободно:", output)


if __name__ == "__main__":
    unittest.main()
