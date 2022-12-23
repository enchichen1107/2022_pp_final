CXXFLAGS := -lm -O3 -std=c++11
EXES     := basic blocked

alls: $(EXES)

clean:
	rm -f $(EXES)

basic: basic.cc basic_lu.cc
	g++ $(CXXFLAGS) -o $@ $?

blocked: blocked.cc blocked_lu.cc
	g++ $(CXXFLAGS) -o $@ $?