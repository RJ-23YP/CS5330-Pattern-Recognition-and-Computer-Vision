IDIR = include
CC = g++
CFLAGS = -I$(IDIR) -I/usr/include/eigen3 -I/usr/include/gtk-3.0 $(shell pkg-config --cflags opencv4) $(shell pkg-config --cflags gtk+-3.0)
LDFLAGS = $(shell pkg-config --libs opencv4) $(shell pkg-config --libs gtk+-3.0) # Add GTK libraries here

ODIR = obj
BDIR = bin

_DEPS = classification.h segmentation.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_SRC = main_recognition.cpp classification.cpp segmentation.cpp
SRC = $(patsubst %,src/%,$(_SRC))

_OBJ = $(patsubst %.cpp,%.o,$(_SRC))
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: src/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BDIR)/recognize: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ $(BDIR)/recognize




