CXXFLAGS := -lm -O3 -std=c++11
EXES     := seq

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: sequential.cc basic_lu.cc
	g++ $(CXXFLAGS) -o $@ $?