add_buddy_model_benchmark(resnet_v2_50-benchmark
  OpenCV
  TOOLDIR ${LLVM_MLIR_BINARY_DIR}
  TOOL mlir-opt
  MLIR ResNet-V2-50.mlir
  BITCODE resnet-default.o
  LIBRARY ResNetDefault
  OPTIONS
    --empty-tensor-to-alloc-tensor 
    --one-shot-bufferize="\"bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map\""
    --canonicalize
    --drop-equivalent-buffer-results --finalizing-bufferize
    --convert-linalg-to-loops 
    --canonicalize 
    --convert-vector-to-scf 
    --convert-scf-to-cf 
    --expand-strided-metadata 
    --lower-affine -convert-arith-to-llvm 
    --convert-vector-to-llvm 
    --finalize-memref-to-llvm 
    --arith-expand 
    --convert-math-to-llvm
    --llvm-request-c-wrappers
    --convert-func-to-llvm
    --reconcile-unrealized-casts
  SOURCE Main.cpp ResNetBenchmark.cpp
)

add_buddy_model_benchmark(resnet_v2_50-benchmark-tpp
  OpenCV
  TOOLDIR ${TPP_MLIR_BINARY_DIR}
  TOOL tpp-opt
  MLIR ResNet-V2-50.mlir
  BITCODE resnet-tpp.o
  LIBRARY ResNetDefaultTpp
  OPTIONS
    --pack-conv2DNhwcHwcf="block-factors=32,32"
    --canonicalize --propagate-pack-and-unpack 
    --canonicalize 
    --constant-fold-pack 
    --interchange-conv-to-expose-matmul 
    --element-wise-fusion 
    --tile-consumer-and-fuse-producers="\"tile-sizes=1,1,1\"" 
    --cse --canonicalize 
    --rewrite-to-brgemm --generalize-tensor-pack-unpack
    --empty-tensor-to-alloc-tensor 
    --one-shot-bufferize="\"bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map\""
    --canonicalize
    --drop-equivalent-buffer-results --finalizing-bufferize
    --convert-linalg-to-loops 
    --canonicalize 
    --convert-vector-to-scf 
    --convert-scf-to-cf 
    --expand-strided-metadata 
    --lower-affine -convert-arith-to-llvm 
    --convert-vector-to-llvm 
    --finalize-memref-to-llvm 
    --arith-expand 
    --convert-math-to-llvm
    --llvm-request-c-wrappers
    --convert-func-to-llvm
    --reconcile-unrealized-casts
  SOURCE Main.cpp ResNetBenchmark.cpp
)
