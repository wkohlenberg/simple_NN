CC		:= gcc
CXX		:= g++

BINDIR		= bin
OUTDIR		= output
SRCDIR		= src

LFLAGS		:= 
CXXFLAGS	:= -std=c++11

main:
	mkdir -p $(BINDIR)
	mkdir -p $(OUTDIR)	
	$(CXX) $(SRCDIR)/main.cpp -o $(BINDIR)/neural -Wall $(CXXFLAGS)

clean:
	rm -rf $(BINDIR)/neural $(OUTDIR)
