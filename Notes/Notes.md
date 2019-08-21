# Notes

## Basic building and running
Building (macOS):

```
./build.sh --build
```

Running tests:
```
./build.sh --test
```
(doesn't appear to run mlas tests, though)

Running mlas test:
```
cd build/Linux/Debug
./onnxruntime_mlas_test
```

Tests: square matrix-matrix multiply up to size 320x320
onnxruntime_mlas_test in build directory

onnxruntime_mlas.cmake -- build info for mlas
onnxruntime_unittests.cmake -- build info for tests

