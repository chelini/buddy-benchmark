add_buddy_model_benchmark(resnet_v2_50-benchmark
  OpenCV
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
