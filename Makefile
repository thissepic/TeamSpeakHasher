appname := TeamSpeakHasher

CXX := g++
CXXFLAGS := -std=c++11 -Wall -Iinclude -DHAVE_OPENCL

LDLIBS := -lOpenCL -lpthread

srcfiles := sha1.cpp IdentityProgress.cpp TunedParameters.cpp Config.cpp \
            DeviceContext.cpp TSHasherContext.cpp OpenCLDevice.cpp BackendFactory.cpp main.cpp
objects := $(patsubst %.cpp, %.o, $(srcfiles))

# Conditional CUDA support
NVCC := $(shell which nvcc 2>/dev/null)
ifdef NVCC
  CXXFLAGS += -DHAVE_CUDA
  LDLIBS += -lcudart
  CUDA_GENCODE := -gencode arch=compute_50,code=sm_50 \
                  -gencode arch=compute_75,code=sm_75 \
                  -gencode arch=compute_86,code=sm_86
  cuda_srcfiles := CUDADevice.cu
  cuda_objects := $(patsubst %.cu, %.cu.o, $(cuda_srcfiles))
else
  cuda_objects :=
endif

all: $(appname)

$(appname): $(objects) $(cuda_objects)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(appname) $(objects) $(cuda_objects) $(LDLIBS)

%.cu.o: %.cu
	$(NVCC) -std=c++11 -DHAVE_CUDA -DHAVE_OPENCL $(CUDA_GENCODE) -Xcompiler -Wall -c $< -o $@

depend: .depend

.depend: $(srcfiles)
	rm -f ./.depend
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;

clean:
	rm -f $(objects) $(cuda_objects)

dist-clean: clean
	rm -f *~ .depend

include .depend
