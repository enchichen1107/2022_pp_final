CXXFLAGS := -lm -O3 -std=c++11
EXES     := lu_basic lu_blocked

alls: $(EXES)

clean:
	rm -f $(EXES)

lu_basic: basic.cc lu_basic.cc
	g++ $(CXXFLAGS) -o $@ $?

lu_blocked: blocked.cc lu_blocked.cc
	g++ $(CXXFLAGS) -o $@ $?