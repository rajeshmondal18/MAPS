MAPS:	MAPS.o
	gcc -g -o MAPS MAPS.c -L/usr/local/lib -lfftw3f_threads -lfftw3f -lm
	rm -rf *.o
	rm -rf *~

MAPS.o:	MAPS.c
	gcc -c -g -I/usr/local/include MAPS.c -L/usr/local/lib


clean:
	rm -rf MAPS
	rm -rf *.o
	rm -rf *~
