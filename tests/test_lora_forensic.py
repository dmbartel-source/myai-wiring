import json
import struct

import tools.lora_forensic as lf


def _write_safetensors(path, header, data):
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(data)


def test_summarize_reads_metadata_and_alpha(tmp_path):
    p = tmp_path / "toy.safetensors"
    header = {
        "__metadata__": {"ss_network_dim": "32", "ss_network_alpha": "1",
                         "ss_output_name": "toy"},
        "lora_unet_x.lora_down.weight": {
            "dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "lora_unet_x.lora_up.weight": {
            "dtype": "F32", "shape": [2, 2], "data_offsets": [16, 32]},
        "lora_unet_x.alpha": {
            "dtype": "F32", "shape": [], "data_offsets": [32, 36]},
    }
    data = b"\x00" * 32 + struct.pack("<f", 1.0)
    _write_safetensors(p, header, data)
    out = lf.summarize(str(p))
    assert out["status"] == "ok"
    assert out["metadata"]["ss_network_dim"] == "32"
    assert out["metadata"]["ss_network_alpha"] == "1"
    assert out["tensor_count"] == 3
    assert out["lora_down_count"] == 1
    assert out["lora_up_count"] == 1
    assert out["alpha"]["count"] == 1
    assert out["alpha"]["min"] == 1.0
    assert len(out["sha256"]) == 64


def test_summarize_missing_file(tmp_path):
    out = lf.summarize(str(tmp_path / "nope.safetensors"))
    assert out["status"] == "missing"
