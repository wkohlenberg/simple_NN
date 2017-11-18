CC		:= gcc
CXX		:= g++

BINDIR		= bin
OUTDIR		= output
SRCDIR		= src

LFLAGS		:= -lm
CXXFLAGS	:= -std=c++11

main:
	mkdir -p $(BINDIR)
	mkdir -p $(OUTDIR)
	$(CXX) $(SRCDIR)/main.cpp $(SRCDIR)/mnist.cpp -o $(BINDIR)/neural -Wall $(CXXFLAGS) $(LFLAGS)

	$(CXX) $(SRCDIR)/createTrainData.cpp -o $(BINDIR)/createTrainData $(CXXFLAGS) $(LFLAGS)

debug:
	mkdir -p $(BINDIR)
	mkdir -p $(OUTDIR)
	$(CXX) $(SRCDIR)/main.cpp $(SRCDIR)/mnist.cpp -o $(BINDIR)/gdb_neural -g -Wall $(CXXFLAGS) $(LFLAGS)

	$(CXX) $(SRCDIR)/createTrainData.cpp -o $(BINDIR)/gdb_createTrainData -g -Wall $(CXXFLAGS) $(LFLAGS)

clean:
	rm -rf $(BINDIR)/neural $(OUTDIR)
