import torch
import socket
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse


def send_tensor(state_dict_key, data, sock, end: bool):
    storage = data.untyped_storage()
    (storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
     ref_counter_handle, ref_counter_offset, event_handle, event_sync_required) = storage._share_cuda_()
    sock.send(pickle.dumps({
        "state_dict_key": state_dict_key,
        "dtype": data.dtype,
        "tensor_size": data.shape,
        "tensor_stride": data.stride(),
        "tensor_offset": data.storage_offset(),  # !Not sure about this one.
        "storage_cls": type(storage),
        "storage_device": storage_device,
        "storage_handle": storage_handle,
        "storage_size_bytes": storage_size_bytes,
        "storage_offset_bytes": storage_offset_bytes,
        "requires_grad": False,
        "ref_counter_handle": ref_counter_handle,
        "ref_counter_offset": ref_counter_offset,
        "event_handle": event_handle,
        "event_sync_required": event_sync_required,
        "end": end,
    }))


def send_state_dict(state_dict, sock, config):
    for i, key in enumerate(state_dict.keys()):
        print(key)
        end = i == len(state_dict.keys()) - 1
        send_tensor(key, state_dict[key], sock, end)
        sock.recv(4096)


def to_tensor_parallel(state_dict, rank, world_size):
    for key in state_dict.keys():
        if 'layernorm' in key:
            continue
        if 'layer_norm' in key:
            continue
        tp_dim = 0
        if 'attention.dense' in key:
            tp_dim = 1
        elif 'dense_4h_to_h' in key:
            tp_dim = 1
        if tp_dim >= len(state_dict[key].shape):
            # can skip
            continue
        state_dict[key] = torch.chunk(state_dict[key], world_size, dim=tp_dim)[rank].contiguous().detach().clone()
    return state_dict


def main(tp_rank, world_size):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 6000 + tp_rank))
    s.listen(1)
    conn, addr = s.accept()
    mdl = AutoModelForCausalLM.from_pretrained(
        "/home/dakota/git-repos/gpt-neox/checkpoints/dpo/pythia/6-9b_step210_hf",
        torch_dtype='auto').to(f"cuda:{tp_rank}")
    state_dict = mdl.state_dict()
    state_dict = to_tensor_parallel(state_dict, tp_rank, world_size)
    send_state_dict(state_dict, conn, mdl.config)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    # setup tp rank and world size
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_rank", type=int, default=0)
    parser.add_argument("--tp_world_size", type=int, default=1)
    args = parser.parse_args()
    main(args.tp_rank, args.tp_world_size)