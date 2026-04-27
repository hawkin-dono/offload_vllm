import h5py
import numpy as np
import os

def print_hdf5_structure(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"=== Inspecting HDF5 file: {filepath} ===")
    
    with h5py.File(filepath, 'r') as f:
        # Hàm in cấu trúc cây đệ quy
        def print_attrs(name, obj):
            # Tính toán khoảng thụt lề dựa trên độ sâu của cây
            level = name.count('/')
            indent = "  " * level
            
            # Tên ngắn gọn của node
            short_name = name.split('/')[-1]
            
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}├── [Dataset] {short_name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}└── [Group] {short_name}")
                
        # # Duyệt toàn bộ cấu trúc file
        f.visititems(print_attrs)


def load_specific_tensor(filepath, seq_id, step, layer_id, tensor_name):
    """
    Hàm tham khảo để load cụ thể 1 tensor lên numpy array
    """
    if not os.path.exists(filepath):
        return

    with h5py.File(filepath, 'r') as f:
        # Đường dẫn cấu trúc: seq_id/step_X/layer_Y/tensor_name
        path = f"{seq_id}/step_{step}/layer_{layer_id}/{tensor_name}"
        
        if path in f:
            data = f[path][:] # [:] để load dữ liệu từ ổ cứng lên RAM
            print(f"\n=== Loaded Tensor: {path} ===")
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Preview (first 5 elements):\n{data.flatten()[:5]}")
            return data
        else:
            print(f"\n[!] Path not found: {path}")
            return None

def print_all_req_ids(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with h5py.File(filepath, 'r') as f:
        req_ids = list(f.keys())
        print(f"\n=== All Request IDs ({len(req_ids)}) ===")
        for req_id in req_ids:
            print(f" - {req_id}")
        return req_ids

def print_all_req_ids_and_steps(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with h5py.File(filepath, 'r') as f:
        req_ids = list(f.keys())
        print(f"\n=== All Request IDs and Steps ===")
        for req_id in req_ids:
            steps = list(f[req_id].keys())
            print(f" - {req_id}: {len(steps)} steps -> {steps}")

if __name__ == "__main__":
    # h5_file = "dataset_generate/dataset_hidden_states.h5"
    h5_file = "vllm_hidden_states.h5"
    
    # 1. In toàn bộ cấu trúc dataset
    # print_hdf5_structure(h5_file)
    
    # 2. In tất cả req_id và step
    print_all_req_ids_and_steps(h5_file)
    
    # 3. Xử lý ví dụ một tensor lấy lên xem thử
    # Thay đổi tham số seq_id, step, layer, tensor_name cho phù hợp với file thực tế
    print("\n# Example: Loading a specific tensor...")
    load_specific_tensor(h5_file, "seq_001", step=50, layer_id="13", tensor_name="router_logits")
    
    with h5py.File(h5_file, 'r') as f:
        print(f.keys())
        print(f["seq_001"].keys())
        print(f["seq_001"]["step_50"].keys())
        print(f["seq_001"]["step_50"]["layer_13"].keys())
        print(f["seq_001"]["step_50"]["layer_13"]["router_logits"][:])