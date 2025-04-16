all: vshow

vshow: vshow.c
	$(CC) vshow.c -O2 -Wall -W -o vshow

clean:
	rm -rf vshow
