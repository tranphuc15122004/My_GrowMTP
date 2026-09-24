import sys
import unittest
from pathlib import Path


SGLANG_PYTHON = Path(__file__).resolve().parents[1] / "sglang/python"
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))

from sglang.srt.server_args import _default_grpc_port


class SGLangGrpcPortTests(unittest.TestCase):
    def test_default_grpc_port_stays_in_tcp_range_for_ephemeral_http_ports(self):
        cases = {
            30000: 40000,
            55535: 65535,
            55536: 45536,
            60766: 50766,
            65535: 55535,
        }
        for http_port, expected in cases.items():
            with self.subTest(http_port=http_port):
                grpc_port = _default_grpc_port(http_port)
                self.assertEqual(grpc_port, expected)
                self.assertGreaterEqual(grpc_port, 1)
                self.assertLessEqual(grpc_port, 65535)
                self.assertNotEqual(grpc_port, http_port)


if __name__ == "__main__":
    unittest.main()
