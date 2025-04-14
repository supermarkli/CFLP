#!/usr/bin/env python3

import os
import subprocess
import sys

def install_dependencies():
    """安装必要的依赖"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "grpcio", "grpcio-tools", "protobuf"])

def generate_grpc_code():
    """生成 gRPC 代码"""
    print("Generating gRPC code...")
    
    # 获取当前脚本所在目录的父目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 设置路径
    proto_dir = os.path.join(base_dir, "proto")
    generated_dir = os.path.join(base_dir, "grpc", "generated")
    
    # 确保目录存在
    if not os.path.exists(proto_dir):
        print(f"Error: proto directory not found at {proto_dir}")
        sys.exit(1)
        
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    
    # 生成 Python gRPC 代码
    subprocess.check_call([
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={generated_dir}",
        f"--grpc_python_out={generated_dir}",
        os.path.join(proto_dir, "federation.proto")
    ])
    
    # 检查生成的文件
    generated_files = [
        os.path.join(generated_dir, "federation_pb2.py"),
        os.path.join(generated_dir, "federation_pb2_grpc.py")
    ]
    
    for file in generated_files:
        if not os.path.exists(file):
            print(f"Error: Failed to generate {file}")
            sys.exit(1)
        
    print("Successfully generated gRPC files:")
    for file in generated_files:
        print(f"- {file}")

if __name__ == "__main__":
    install_dependencies()
    generate_grpc_code() 