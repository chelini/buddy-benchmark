add_buddy_model_benchmark(resnet_v2_50-benchmark-xsmm
  OpenCV
  TOOLDIR ${TPP_MLIR_BINARY_DIR}
  TOOL tpp-opt
  MLIR ResNet-V2-50.mlir
  BITCODE resnet-xsmm.o
  LIBRARY ResNetDefaultXsmm
  OPTIONS
    --pack-conv2DNhwcHwcf="\"block-factors=32,32\""
    --propagate-pack-and-unpack 
    --canonicalize 
    --constant-fold-pack 
    --element-wise-fusion 
    --tile-consumer-and-fuse-producers="\"tile-sizes=1,1,1\"" 
    --cse --canonicalize 
    --interchange-conv-to-expose-matmul
    --rewrite-conv-to-matmul-or-brgemm 
    --generalize-tensor-pack-unpack
    --empty-tensor-to-alloc-tensor 
    --bufferize
    --convert-linalg-to-tpp="\"use-parallel-loops=false\""
    -convert-linalg-to-loops --canonicalize 
    --convert-tpp-to-xsmm 
    --loop-invariant-code-motion 
    --convert-xsmm-to-func 
    --convert-vector-to-scf 
    --convert-scf-to-cf 
    --expand-strided-metadata 
    --lower-affine 
    --convert-arith-to-llvm 
    --convert-vector-to-llvm 
    --finalize-memref-to-llvm 
    --arith-expand 
    --convert-math-to-llvm 
    --llvm-request-c-wrappers 
    --convert-func-to-llvm --canonicalize 
    --reconcile-unrealized-casts 
  SOURCE Main.cpp ResNetBenchmark.cpp
)
