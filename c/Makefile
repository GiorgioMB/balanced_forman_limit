CFLAGS = -O3 -march=native
PROGS = d2c fullgraph fullvar

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(PROGS)
